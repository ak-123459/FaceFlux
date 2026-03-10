"""
inference_api.py  —  Face Inference API  (inference-only, no local FAISS)
────────────────────────────────────────────────────────────────────────────
This machine ONLY handles:
  • ONNX embedding inference
  • Calling the remote Vector Store API for matching / storing

All FAISS operations are delegated to the Vector Store API (machine-c).

Endpoints
─────────
  POST /attendance/batch      N face crops → embed → remote match → DB insert
  POST /faces/register        N face crops → avg embed → remote store
  DELETE /faces/{user_id}     Proxy delete to vector store API
  GET  /attendance            Query attendance records
  GET  /health                Liveness + stats
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Optional
import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv

from pipelines.onnx_embed_engine import OnnxEmbedEngine
from pipelines.batch_pipeline import FaceRecognitionPipeline, BatchResult, _decode
from services.vector_store_client import VectorStoreClient
from database.database import (
    DatabaseConfig, UserManager,
    AttendanceManager, init_database,
)
from services.data_validation.validation import (
    HealthOut, BatchOut, FrameOut,
    RegisterOut, DeleteOut,
)
import numpy as np

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt = "%H:%M:%S")


logger = logging.getLogger("inference_api")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class _Cfg:
    EMB_MODEL            = os.getenv("EMB_MODEL_PATH",        "models/w600k_r50_int8.onnx")
    EMB_MODEL_TYPE = os.getenv("EMB_MODEL_TYPE", "arcface")
    VECTOR_DB_HOST       = os.getenv("VECTOR_DB_HOST",        "localhost")
    VECTOR_DB_PORT       = os.getenv("VECTOR_DB_PORT",        "8005")
    VECTOR_STORE_TIMEOUT = float(os.getenv("VECTOR_STORE_TIMEOUT", "5.0"))
    CAMERA_ID            = os.getenv("CAMERA_ID",             None)
    MAX_BATCH            = int(os.getenv("MAX_BATCH_SIZE",    "64"))

    HOST    = os.getenv("EMB_MODEL_HOST",    "0.0.0.0")
    PORT    = int(os.getenv("EMB_MODEL_PORT",   "8004"))
    WORKERS = int(os.getenv("EMB_NUM_WORKER",   "1"))

    # Thread pool for all blocking work
    # (ONNX inference, DB calls, image decode)
    THREAD_WORKERS = int(os.getenv("THREAD_POOL_SIZE", "4"))


# ─────────────────────────────────────────────────────────────────────────────
# Shared thread pool  (used by run_in_executor throughout)
# ─────────────────────────────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(
    max_workers        = _Cfg.THREAD_WORKERS,
    thread_name_prefix = "inference_worker",
)


async def _run(fn, *args):
    """Run a blocking function in the shared thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan  (startup + shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────────────────

    # 1. DB init — blocking, run in executor
    logger.info("Init DB …")
    await _run(init_database)
    logger.info("DB initialised.")

    # 2. ONNX model — blocking load, run in executor
    logger.info("Loading ONNX embedding model …")
    engine = await _run(OnnxEmbedEngine.instance,_Cfg.EMB_MODEL,None, _Cfg.EMB_MODEL_TYPE)
    logger.info("ONNX model ready.")

    # 3. Vector store client — async HTTP client, no blocking
    logger.info("Connecting to Vector Store API → %s", _Cfg.VECTOR_DB_HOST)
    store = VectorStoreClient(
        base_url = f"http://{_Cfg.VECTOR_DB_HOST}:{_Cfg.VECTOR_DB_PORT}",
        timeout  = _Cfg.VECTOR_STORE_TIMEOUT,
    )

    # 4. Health check vector store
    try:
        h = await store.health()
        logger.info(
            "✓ Vector Store API reachable  (registered faces: %d)",
            h.get("vectors_stored", "?"),
        )
    except Exception as exc:
        logger.warning("⚠ Vector Store API not reachable at startup: %s", exc)

    # 5. Build pipeline — stateless, safe to share across requests
    pipeline = FaceRecognitionPipeline(
        engine    = engine,
        store     = store,
        camera_id = _Cfg.CAMERA_ID,
    )

    # 6. Store on app.state — accessible in every route via request.app.state
    app.state.pipeline = pipeline
    app.state.store    = store
    app.state.engine   = engine

    logger.info("✓ Inference pipeline ready.")

    # ── hand control to the app ───────────────────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    await store.close()                      # close httpx async client
    _executor.shutdown(wait=False)           # stop thread pool
    logger.info("Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Face Inference API",
    description = "Inference-only node — delegates vector storage to remote FAISS API",
    version     = "2.1.0",
    lifespan    = lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=512)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pl(request: Request) -> FaceRecognitionPipeline:
    """Get the shared pipeline from app.state."""
    pl = getattr(request.app.state, "pipeline", None)
    if pl is None:
        raise HTTPException(503, "Pipeline not ready.")
    return pl


async def _read(files: list[UploadFile]) -> tuple[list[bytes], list[str]]:
    """
    Read all uploaded files concurrently.
    Each f.read() is an async I/O call so we gather them in parallel.
    """
    async def _read_one(f: UploadFile) -> tuple[bytes, str]:
        data = await f.read()
        return data, f.filename or ""

    pairs = await asyncio.gather(*[_read_one(f) for f in files])
    raws  = [p[0] for p in pairs]
    names = [p[1] for p in pairs]
    return raws, names


def _to_out(br: BatchResult) -> BatchOut:
    return BatchOut(
        total          = br.total,
        matched        = br.matched,
        attendance_new = br.attendance_new,
        errors         = br.errors,
        elapsed_ms     = round(br.elapsed_ms, 1),
        frames=[
            FrameOut(
                frame_index    = r.frame_index,
                filename       = r.filename,
                matched        = r.matched,
                user_id        = r.user_id,
                similarity     = round(r.similarity, 4),
                f = r.attendance_new,
                error          = r.error,
            )
            for r in br.frames
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthOut, tags=["System"])
async def health(request: Request):
    """Liveness + stats check."""

    # DB ping — blocking, run in executor
    db_ok = await _run(lambda: DatabaseConfig.get_connection() is not None)

    # Vector store health — async HTTP, no executor needed
    vector_store_ok = False
    registered      = 0
    try:
        h               = await request.app.state.store.health()
        vector_store_ok = True
        registered      = h.get("vectors_stored", 0)
    except Exception as exc:
        logger.warning("health: vector store unreachable: %s", exc)

    return HealthOut(
        status           = "ok",
        registered_faces = registered,
        db_ok            = db_ok,
        vector_store_url = _Cfg.VECTOR_DB_HOST,
        vector_store_ok  = vector_store_ok,
    )


@app.post("/admin/switch-model", tags=["Admin"])
async def switch_model(
    request: Request,
    model_path: str = Form(...),
    model_type: str = Form("arcface"),   # "arcface" | "vggface2-face"
):
    engine = await _run(OnnxEmbedEngine.instance, model_path, None, model_type)
    request.app.state.pipeline._engine = engine
    request.app.state.engine = engine
    return {"switched_to": model_path, "model_type": model_type}





# ── Batch attendance ──────────────────────────────────────────────────────────

@app.post(
    "/attendance/batch",
    response_model = BatchOut,
    tags           = ["Attendance"],
    summary        = "Bulk attendance — send pre-cropped face images",
)
async def attendance_batch(
    request:   Request,
    files:     Annotated[list[UploadFile], File(description="Pre-cropped face images (JPEG/PNG)")],
    camera_id: Annotated[Optional[str], Form()] = None,
):
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > _Cfg.MAX_BATCH:
        raise HTTPException(413, f"Max batch size is {_Cfg.MAX_BATCH}.")

    # ── read all files concurrently ───────────────────────────────────────
    raws, names = await _read(files)
    logger.info(
        "attendance_batch: received files=%d  camera_id=%s",
        len(files), camera_id,
    )

    # ── run pipeline ──────────────────────────────────────────────────────
    # FaceRecognitionPipeline is stateless — safe to call concurrently.
    # Pass camera_id directly instead of mutating shared state.
    pl = _pl(request)
    try:
        result = await pl.process_batch(
            raws,
            filenames = names,
            camera_id = camera_id or pl.camera_id,
        )
    except Exception as exc:
        logger.error("attendance_batch pipeline error: %s", exc)
        raise HTTPException(502, f"Pipeline error: {exc}")

    logger.info(
        "attendance_batch: done  matched=%d  new=%d  errors=%d  %.1fms",
        result.matched, result.attendance_new,
        result.errors,  result.elapsed_ms,
    )
    return _to_out(result)


# ── Register face ─────────────────────────────────────────────────────────────
@app.post("/faces/register", response_model=RegisterOut, tags=["Faces"])
async def register_face(
    request:    Request,
    files:      Annotated[list[UploadFile], File()],
    name:       Annotated[str,  Form()] = "Unknown",
    email:      Annotated[str,  Form()] = None,
    phone:      Annotated[str,  Form()] = None,
    department: Annotated[str,  Form()] = "Unknown",
    role:       Annotated[str,  Form()] = "Employee",
    threshold:  Annotated[float,Form()] = 0.80,
):
    if not files:
        raise HTTPException(400, "No image files provided.")

    # ── Step 1: Read + decode images ──────────────────────────────────────
    raws, _ = await _read(files)
    if not raws:
        raise HTTPException(400, "No valid image files.")

    pl = _pl(request)

    # ── Step 2: Create embedding ──────────────────────────────────────────
    try:
        decoded = await asyncio.gather(*[_run(_decode, raw) for raw in raws])
        crops   = [img for img in decoded if img is not None]
        if not crops:
            raise HTTPException(400, "No valid faces could be decoded from images.")

        embeddings = await _run(pl._engine.embed_batch, crops)
        avg        = embeddings.mean(axis=0).astype(np.float32)
        norm       = np.linalg.norm(avg)
        avg        = avg / norm if norm > 1e-6 else avg

        logger.info("register_face: embedding created  shape=%s", avg.shape)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("register_face: embedding failed: %s", exc)
        raise HTTPException(500, f"Embedding failed: {exc}")

    # ── Step 3: Duplicate check against vector store ──────────────────────
    try:
        duplicate_id, similarity = await pl._store.search_one(avg)

        logger.info(
            "register_face: duplicate_check  best_match=%s  similarity=%.4f  threshold=%.4f",
            duplicate_id, similarity, threshold)

        if duplicate_id and similarity >= threshold:
            # Fetch existing user details for helpful error response
            existing_user = await _run(UserManager.get_user, duplicate_id)
            raise HTTPException(409, {
                "error":       "Duplicate face detected.",
                "similarity":  round(similarity * 100, 2),
                "threshold":   round(threshold * 100, 2),
                "existing_user": {
                    "user_id":    duplicate_id,
                    "name":       existing_user.get("name")       if existing_user else "Unknown",
                    "department": existing_user.get("department") if existing_user else "Unknown",
                    "role":       existing_user.get("role")       if existing_user else "Unknown",
                }
            })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("register_face: duplicate check failed: %s", exc)
        raise HTTPException(500, f"Duplicate check failed: {exc}")

    try:
        user_id = await _run(UserManager.add_user_auto, name, email, phone, department, role)
        if not user_id:
            raise HTTPException(500, "Failed to create user in database.")
        logger.info("register_face: user created user_id=%s name=%s", user_id, name)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("register_face: DB insert failed: %s", exc)
        raise HTTPException(500, f"Database insert failed: {exc}")

    # ── Step 5: Store embedding in vector store ───────────────────────────
    try:
        ok, msg = await pl._store.add(str(user_id), avg)
        if not ok:
            await _run(UserManager.delete_user, user_id)  # ✅ INT not str
            raise HTTPException(500, f"Vector store failed: {msg}. Rolled back.")
        logger.info("register_face: embedding stored user_id=%s", user_id)

    except HTTPException:
        raise
    except Exception as exc:
        await _run(UserManager.delete_user, user_id)  # ✅ INT not str
        logger.error("register_face: vector store error: %s", exc)
        raise HTTPException(502, f"Vector store unreachable: {exc}. Rolled back.")


    # ── Step 6: Return success ────────────────────────────────────────────
    logger.info("register_face: SUCCESS  user_id=%s  name=%s", user_id, name)
    return RegisterOut(success=True, user_id=str(user_id), message="User registered successfully.")



# ── Delete face ───────────────────────────────────────────────────────────────

@app.delete("/faces/{user_id}", response_model=DeleteOut, tags=["Faces"])
async def delete_face(user_id: str, request: Request):
    """Proxy delete → Vector Store API."""
    logger.info("delete_face: user_id=%s", user_id)
    try:
        # store.delete is async HTTP call
        ok, msg = await request.app.state.store.delete(user_id)
    except Exception as exc:
        logger.error("delete_face error: %s", exc)
        raise HTTPException(502, f"Vector store unreachable: {exc}")

    if not ok:
        raise HTTPException(404, msg)

    logger.info("delete_face: deleted user_id=%s", user_id)
    return DeleteOut(success=True, message=msg)


# ── Attendance query ──────────────────────────────────────────────────────────

@app.get("/attendance", tags=["Attendance"], summary="Query attendance records")
async def get_attendance(
    user_id:    Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date:   Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    logger.info(
        "get_attendance: user_id=%s  start=%s  end=%s",
        user_id, start_date, end_date,
    )

    # AttendanceManager.get_attendance_records is blocking DB call
    # run in executor so event loop stays free
    records = await _run(
        AttendanceManager().get_attendance_records,
        start_date, end_date, user_id,
    )

    logger.info("get_attendance: returned %d records", len(records))
    return {"total": len(records), "records": records}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
