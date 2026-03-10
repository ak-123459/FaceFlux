"""
main.py  —  FAISS Vector Store API
────────────────────────────────────────────────────────────────────────────
Receives embeddings from your inference API and performs:
  • /register   — add a new user embedding
  • /upsert     — add or replace user embedding
  • /match      — match single embedding  (verification / 1:1)
  • /search     — search best match from store  (identification / 1:N)
  • /batch      — batch search N embeddings in one call
  • /delete     — remove user
  • /exists     — check if user registered
  • /stats      — store info

Run:
    uvicorn main:app --host 0.0.0.0 --port 8002 --workers 4
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from database.vector_store_db import VectorStore, DIM
from services.data_validation.validation import EmbeddingIn,QueryIn,VerifyIn,BatchQueryIn
from dotenv import load_dotenv





load_dotenv()



# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("faiss_api")


HOST = os.getenv("VECTOR_DB_HOST")
PORT = int(os.getenv("VECTOR_DB_PORT"))
WORKER = int(os.getenv("VECTOR_DB_WORKER"))

# ─────────────────────────────────────────────────────────────────────────────
# App & Store
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FAISS Vector Store API",
    description="Receives face embeddings from inference API → stores / searches / matches",
    version="1.0.0",
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



store = VectorStore(
    index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss.index"),
    meta_path=os.getenv("FAISS_META_PATH",   "data/faiss_meta.pkl"),
    threshold=float(os.getenv("MATCH_THRESHOLD", "0.50")),
    top_k=int(os.getenv("TOP_K", "1")),
)

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(embedding: list[float]) -> np.ndarray:
    return np.array(embedding, dtype=np.float32)


def _match_result_to_dict(result) -> Optional[dict]:
    if result is None:
        return None
    return {
        "user_id":    result.user_id,
        "distance":   round(result.distance, 6),
        "similarity": round(result.similarity, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "vectors_stored": await store.get_size()}


@app.get("/stats")
async def stats():
    """Return current store statistics."""
    return {
        "vectors_stored": await store.get_size(),
        "embedding_dim":  DIM,
        "threshold":      float(os.getenv("MATCH_THRESHOLD", "0.40")),
    }


# ── Register (add new user, reject if exists) ─────────────────────────────

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: EmbeddingIn):
    """
    Add a new user embedding.
    Returns 409 if user_id already registered.

    Called by: enrollment pipeline after inference API returns embedding.
    """
    t0 = time.perf_counter()
    ok, msg = await store.add(body.user_id, _to_np(body.embedding))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not ok:
        raise HTTPException(status_code=409, detail=msg)

    return {
        "status":     "registered",
        "user_id":    body.user_id,
        "total":      await store.get_size(),
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ── Upsert (add or replace) ───────────────────────────────────────────────

@app.post("/upsert")
async def upsert(body: EmbeddingIn):
    """
    Add or replace embedding for user_id.
    Use this when user re-enrolls (new photo, updated model, etc.)
    """
    t0 = time.perf_counter()
    ok, msg = store.upsert(body.user_id, _to_np(body.embedding))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not ok:
        raise HTTPException(status_code=500, detail=msg)

    return {
        "status":     "upserted",
        "user_id":    body.user_id,
        "total":      await store.get_size(),
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ── Search (1:N identification) ───────────────────────────────────────────

@app.post("/search")
async def search(body: QueryIn):
    """
    Search the store for best match.
    Returns matched user or null if below threshold.

    Called by: face recognition pipeline after inference API returns embedding.
    """
    t0 = time.perf_counter()
    results = await store.search_batch_async(_to_np(body.embedding))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    match = _match_result_to_dict(results[0]) if results else None

    return {
        "matched":    match is not None,
        "match":      match,
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ── Verify (1:1 verification — is this user who they claim to be?) ────────

@app.post("/verify")
async def verify(body: VerifyIn):
    """
    Check if the given embedding matches a specific user_id.
    1:1 verification (not open-set search).

    Example use: door access — "is this person user_id=john_doe?"
    """
    if not await store.user_exists(body.user_id):
        raise HTTPException(status_code=404, detail=f"User '{body.user_id}' not registered")

    t0 = time.perf_counter()
    results = await store.search_batch_async(_to_np(body.embedding))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    match = results[0] if results else None
    verified = match is not None and match.user_id == body.user_id

    return {
        "verified":   verified,
        "user_id":    body.user_id,
        "match":      _match_result_to_dict(match),
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ── Batch search (N embeddings in one call) ───────────────────────────────

@app.post("/batch")
async def batch_search(body: BatchQueryIn):
    """
    Search N embeddings in a single FAISS call.
    Much faster than N sequential /search calls.

    Use for: processing multiple faces detected in one frame.
    """
    t0 = time.perf_counter()
    arr = np.array(body.embeddings, dtype=np.float32)
    results = await store.search_batch(arr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    matches = [_match_result_to_dict(r) for r in results]

    return {
        "count":      len(matches),
        "results":    matches,
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ── Delete ────────────────────────────────────────────────────────────────

@app.delete("/delete/{user_id}")
async def delete(user_id: str):
    """Remove a user's embedding from the store."""
    ok, msg = await store.delete(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail=msg)

    return {
        "status":  "deleted",
        "user_id": user_id,
        "total":   store.size,
    }


# ── Exists ────────────────────────────────────────────────────────────────

@app.get("/exists/{user_id}")
async def exists(user_id: str):
    """Check if a user is already registered."""
    return {
        "user_id": user_id,
        "exists":   await store.user_exists(user_id),
    }


# ── Save (manual flush) ───────────────────────────────────────────────────

@app.post("/save")
async def save():
    """Manually flush index to disk (auto-saved on every write)."""
    ok = await store.save()
    if not ok:
        raise HTTPException(status_code=500, detail="Save failed")
    return {"status": "saved", "total": store.size}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.vector_services_api:app", host=HOST, port=PORT, workers=WORKER)
