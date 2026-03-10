"""
batch_pipeline.py
────────────────────────────────────────────────────────────────────────────
Pipeline for pre-cropped, pre-aligned face images.
No detection. No landmark alignment.

Flow:
  process_batch()
    1. decode bytes → numpy  (parallel via asyncio.gather + run_in_executor)
    2. embed_batch()         → (N, 512)  [single ONNX call, run_in_executor]
    3. search_batch()        → matches   [single async HTTP/FAISS call]
    4. already_marked()      → set[str]  [DB check, run_in_executor]
    5. write_batch()         → INSERT    [executemany, run_in_executor]

  register_face()
    1. decode bytes → numpy  (parallel, run_in_executor)
    2. embed_batch()         → average → normalise  [run_in_executor]
    3. await store.add() / upsert()
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from pipelines.onnx_embed_engine import OnnxEmbedEngine
from services.vector_store_client import VectorStoreClient, MatchResult  # ✅ correct import
from database.database import DatabaseConfig, AttendanceStatus
from services.data_validation.validation import FrameResult, BatchResult

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB hard cap per image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode(raw: bytes) -> Optional[np.ndarray]:
    """bytes → BGR numpy array. None on failure or oversize."""
    if len(raw) > MAX_IMAGE_BYTES:
        logger.warning("Image %d bytes exceeds 5MB cap — skipped.", len(raw))
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("cv2.imdecode returned None — corrupt or unsupported image.")
    return img


def _attendance_status(now: datetime) -> str:
    """Present before 09:15, Late after. Edit to match your policy."""
    if now.hour > 9 or (now.hour == 9 and now.minute > 15):
        return AttendanceStatus.LATE
    return AttendanceStatus.PRESENT


async def _run(fn, *args):
    """Run a blocking callable in the default thread pool — non-blocking."""
    loop = asyncio.get_running_loop()   # ✅ correct for Python 3.10+
    return await loop.run_in_executor(None, fn, *args)


# ─────────────────────────────────────────────────────────────────────────────
# Attendance writer  (sync methods — always called via _run)
# ─────────────────────────────────────────────────────────────────────────────

class _AttendanceWriter:

    # ✅ MariaDB syntax (not SQLite)
    _SQL = """
        INSERT IGNORE INTO attendance
            (user_id, date, time, status, confidence_score, camera_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    def already_marked(self, user_ids: list[str], today: str) -> set[str]:
        """Sync DB read. Always call via: await _run(writer.already_marked, ...)"""
        logger.debug(
            "already_marked: checking %d users for date=%s", len(user_ids), today
        )
        if not user_ids:
            return set()

        conn = DatabaseConfig.get_connection()
        if not conn:
            logger.error("already_marked: no DB connection.")
            return set()

        # ✅ MariaDB placeholders — ["%s", "%s"] not "??"
        ph = ",".join(["%s"] * len(user_ids))
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                f"SELECT user_id FROM attendance WHERE date=%s AND user_id IN ({ph})",
                [today, *user_ids],
            )
            marked = {r["user_id"] for r in cur.fetchall()}
            logger.debug(
                "already_marked: %d already marked today → %s", len(marked), marked
            )
            return marked
        except Exception as exc:
            logger.error("already_marked failed: %s", exc)
            return set()
        finally:
            try: cur.close(); conn.close()
            except Exception: pass


    def write_batch(self, rows: list[tuple]) -> int:
        """Sync DB write. Always call via: await _run(writer.write_batch, rows)"""
        logger.info("write_batch: inserting %d rows.", len(rows))
        for i, row in enumerate(rows):
            logger.debug("  row[%d]: %s", i, row)

        if not rows:
            return 0

        conn = DatabaseConfig.get_connection()
        if not conn:
            logger.error("write_batch: no DB connection.")
            return 0
        try:
            cur = conn.cursor()
            cur.executemany(self._SQL, rows)
            inserted = max(cur.rowcount, 0)
            conn.commit()
            logger.info("write_batch: committed inserted=%d", inserted)
            return inserted
        except Exception as exc:
            logger.error("write_batch failed: %s  rows=%s", exc, rows)
            conn.rollback()
            return 0
        finally:
            try: cur.close(); conn.close()
            except Exception: pass



# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FaceRecognitionPipeline:
    """
    Stateless async pipeline — safe to share across all concurrent requests.

    Every blocking operation (ONNX inference, DB calls, image decode) runs
    via run_in_executor so the asyncio event loop is never stalled.
    """

    def __init__(
        self,
        engine:    OnnxEmbedEngine,
        store:     VectorStoreClient,   # ✅ correct type hint
        camera_id: Optional[str] = None,
    ) -> None:
        self._engine   = engine
        self._store    = store
        self._writer   = _AttendanceWriter()
        self.camera_id = camera_id
        logger.info("FaceRecognitionPipeline ready  camera_id=%s", camera_id)

    # ── batch attendance ─────────────────────────────────────────────────────

    async def process_batch(
        self,
        images:    list[bytes],
        filenames: Optional[list[str]] = None,
        camera_id: Optional[str]       = None,   # ✅ per-request override
    ) -> BatchResult:
        """
        Mark attendance for a batch of pre-cropped face images.

        Parameters
        ──────────
        images    : list of raw JPEG/PNG bytes, one per person.
        filenames : optional parallel list of filenames (for logging/response).
        camera_id : per-request camera_id override (avoids mutating shared state).
        """
        t0               = time.perf_counter()
        n                = len(images)
        effective_camera = camera_id or self.camera_id
        results          = [FrameResult(frame_index=i) for i in range(n)]
        br               = BatchResult(total=n)

        logger.info(
            "process_batch: START total=%d camera_id=%s", n, effective_camera
        )

        if filenames:
            for i, fn in enumerate(filenames[:n]):
                results[i].filename = fn

        # ── 1. parallel decode ────────────────────────────────────────────
        logger.debug("process_batch: [1/5] decoding %d images in parallel …", n)
        t_dec   = time.perf_counter()
        decoded: list[Optional[np.ndarray]] = await asyncio.gather(*[
            _run(_decode, raw) for raw in images   # ✅ all decoded in parallel
        ])
        logger.info(
            "process_batch: [1/5] decode done  %.1fms",
            (time.perf_counter() - t_dec) * 1000,
        )

        crops:    list[np.ndarray] = []
        crop_map: list[int]        = []

        for i, img in enumerate(decoded):
            if img is None:
                logger.warning("process_batch: frame[%d] decode failed.", i)
                results[i].error = "decode_failed"
                br.errors += 1
                continue
            crops.append(img)
            crop_map.append(i)

        if not crops:
            logger.warning("process_batch: all images failed decode — aborting.")
            br.elapsed_ms = (time.perf_counter() - t0) * 1000
            br.frames     = results
            return br

        # ── 2. ONNX embed — non-blocking via executor ─────────────────────
        logger.debug("process_batch: [2/5] embedding %d crops …", len(crops))
        t_emb      = time.perf_counter()
        embeddings = await _run(self._engine.embed_batch, crops)   # ✅ non-blocking
        logger.info(
            "process_batch: [2/5] embed done  shape=%s  %.1fms  per_img=%.1fms",
            embeddings.shape,
            (time.perf_counter() - t_emb) * 1000,
            (time.perf_counter() - t_emb) * 1000 / len(crops),
        )

        # ── 3. FAISS search — async HTTP (non-blocking) ───────────────────
        logger.debug("process_batch: [3/5] FAISS search n=%d …", len(crops))
        t_search = time.perf_counter()
        matches: list[Optional[MatchResult]] = await self._store.search_batch(embeddings)
        logger.info(
            "process_batch: [3/5] search done  matched=%d  %.1fms",
            sum(1 for m in matches if m is not None),
            (time.perf_counter() - t_search) * 1000 )
        for j, m in enumerate(matches):
            if m:
                logger.debug(
                    "  match[%d]: user_id=%s dist=%.4f sim=%.4f",
                    j, m.user_id, m.distance, m.similarity,
                )

        # ── 4. DB already_marked check — non-blocking via executor ────────
        now         = datetime.now()
        today       = now.date().isoformat()
        time_str    = now.strftime("%H:%M:%S")
        status      = _attendance_status(now)
        matched_ids = [m.user_id for m in matches if m is not None]

        logger.debug(
            "process_batch: [4/5] already_marked check  date=%s  n=%d",
            today, len(matched_ids),
        )
        already_marked: set[str] = await _run(   # ✅ non-blocking
            self._writer.already_marked, matched_ids, today
        )
        logger.info(
            "process_batch: [4/5] already_marked=%d", len(already_marked)
        )

        # ── resolve attendance rows ───────────────────────────────────────
        attendance_rows: list[tuple] = []
        seen_this_batch: set[str]    = set()

        for j, match in enumerate(matches):
            frame_idx = crop_map[j]
            if match is None:
                results[frame_idx].error = "no_match"
                continue

            uid = match.user_id
            results[frame_idx].matched    = True
            results[frame_idx].user_id    = uid
            results[frame_idx].similarity = match.similarity
            br.matched += 1

            logger.debug(
                "process_batch: frame[%d] matched user_id=%s sim=%.4f",
                frame_idx, uid, match.similarity)

            if uid in already_marked:

                logger.debug("user_id=%s already marked today — skip.", uid)

            elif uid in seen_this_batch:
                logger.debug("user_id=%s dup in this batch — skip.", uid)

            else:
                attendance_rows.append((
                    uid, today, time_str, status,
                    float(match.similarity), effective_camera))
                seen_this_batch.add(uid)

        logger.info(
            "process_batch: queued %d attendance rows", len(attendance_rows)
        )

        # ── 5. DB write — non-blocking via executor ───────────────────────
        if attendance_rows:
            inserted = await _run(self._writer.write_batch, attendance_rows)  # ✅ non-blocking
            br.attendance_new = inserted
            new_uids = {r[0] for r in attendance_rows}
            for res in results:
                if res.user_id in new_uids:
                    res.attendance_new = True
            logger.info("process_batch: [5/5] inserted=%d", inserted)
        else:
            logger.info("process_batch: [5/5] no new rows to write.")

        br.errors    += sum(1 for r in results if r.error)
        br.elapsed_ms = (time.perf_counter() - t0) * 1000
        br.frames     = results

        logger.info(
            "process_batch: DONE total=%d matched=%d new=%d errors=%d %.1fms",
            n, br.matched, br.attendance_new, br.errors, br.elapsed_ms,
        )
        return br

    # ── face registration ────────────────────────────────────────────────────

    async def register_face(   # ✅ now async — was sync before
        self,
        user_id: str,
        images:  list[bytes],
        upsert:  bool = False,
    ) -> tuple[bool, str]:
        """
        Compute and store a face embedding for user_id.
        Pass multiple images of the same person to average embeddings.

        Parameters
        ──────────
        user_id : must already exist in users table (unless upsert=True)
        images  : 1-N pre-cropped face image bytes
        upsert  : overwrite existing embedding if True
        """
        logger.info(
            "register_face: START user_id=%s images=%d upsert=%s",
            user_id, len(images), upsert,
        )

        # ✅ parallel decode
        decoded = await asyncio.gather(*[_run(_decode, raw) for raw in images])
        crops   = [img for img in decoded if img is not None]

        logger.info(
            "register_face: decoded %d/%d images ok", len(crops), len(images)
        )

        if not crops:
            logger.error("register_face: no valid images decoded.")
            return False, "No valid images could be decoded."

        # ✅ embed in executor — non-blocking
        embeddings = await _run(self._engine.embed_batch, crops)
        avg        = embeddings.mean(axis=0).astype(np.float32)
        norm       = np.linalg.norm(avg)
        avg        = avg / norm if norm > 1e-6 else avg

        logger.debug(
            "register_face: avg embedding norm_after=%.4f", float(np.linalg.norm(avg))
        )

        # ✅ async HTTP call to vector store
        if upsert:
            ok, msg = await self._store.upsert(user_id, avg)
        else:
            ok, msg = await self._store.add(user_id, avg)

        logger.info(
            "register_face: DONE user_id=%s ok=%s msg=%s", user_id, ok, msg
        )
        return ok, msg