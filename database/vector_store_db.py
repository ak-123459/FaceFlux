"""
vector_store.py
────────────────────────────────────────────────────────────────────────────
Thread-safe FAISS wrapper.

Design goals
  • All mutations serialised through a single RW-lock so concurrent reads
    never block each other.
  • Batch search in one FAISS call → far more efficient than N sequential
    searches.
  • Incremental add/delete without reloading the whole index from disk
    (delete rebuilds from in-memory vectors, not from disk).
  • Index flushed to disk only on explicit save() or on safe shutdown.
  • Fully async public interface — safe to call from FastAPI / asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

DIM = 512   # ArcFace / buffalo_s embedding dimension


# ─────────────────────────────────────────────────────────────────────────────
# Read-Write lock
# ─────────────────────────────────────────────────────────────────────────────

class _RWLock:
    """
    Simple readers-writers lock.
    Multiple readers allowed concurrently; writers are exclusive.
    """

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers    = 0

    # ── context-manager helpers ───────────────────────────────────────────────

    class _ReadCtx:
        def __init__(self, rw: "_RWLock") -> None: self._rw = rw
        def __enter__(self):  self._rw._acquire_read();  return self
        def __exit__(self, *_): self._rw._release_read()

    class _WriteCtx:
        def __init__(self, rw: "_RWLock") -> None: self._rw = rw
        def __enter__(self):  self._rw._acquire_write(); return self
        def __exit__(self, *_): self._rw._release_write()

    def reading(self) -> _ReadCtx:  return self._ReadCtx(self)
    def writing(self) -> _WriteCtx: return self._WriteCtx(self)

    def _acquire_read(self) -> None:
        with self._read_ready:
            self._readers += 1

    def _release_read(self) -> None:
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def _acquire_write(self) -> None:
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def _release_write(self) -> None:
        self._read_ready.release()


# ─────────────────────────────────────────────────────────────────────────────
# Match result
# ─────────────────────────────────────────────────────────────────────────────

class MatchResult:
    __slots__ = ("user_id", "distance", "similarity")

    def __init__(self, user_id: str, distance: float) -> None:
        self.user_id    = user_id
        self.distance   = float(distance)
        # L2-distance on unit vectors: d² = 2 − 2·cos  →  cos = 1 − d²/2
        self.similarity = float(1.0 - distance ** 2 / 2.0)

    def __repr__(self) -> str:
        return f"MatchResult({self.user_id!r}, sim={self.similarity:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thread-safe, disk-persisted FAISS vector store with a fully async
    public interface.

    All blocking work (FAISS search, disk I/O, lock acquisition) runs in a
    dedicated ThreadPoolExecutor so the asyncio event loop is never stalled.

    Public async interface
    ──────────────────────
    await store.add(user_id, embedding)       → (bool, str)
    await store.upsert(user_id, embedding)    → (bool, str)
    await store.delete(user_id)               → (bool, str)
    await store.search_batch(embeddings)      → list[MatchResult | None]
    await store.user_exists(user_id)          → bool
    await store.get_size()                    → int
    await store.save()                        → bool
          store.close()                       (sync, call on shutdown)

    Parameters
    ──────────
    index_path  : path to .faiss file
    meta_path   : path to .pkl   file  (list[str] of user_ids)
    threshold   : L2-distance threshold for a positive match
                  cos ≈ 0.80  →  L2 ≈ 0.63 for unit vectors
    top_k       : candidates retrieved from FAISS (≥ 1)
    workers     : thread-pool size
    """

    def __init__(
        self,
        index_path: str   = "data/faiss.index",
        meta_path:  str   = "data/faiss_meta.pkl",
        threshold:  float = 0.63,
        top_k:      int   = 1,
        workers:    int   = 4,
    ) -> None:
        self._index_path = Path(index_path)
        self._meta_path  = Path(meta_path)
        self._threshold  = threshold
        self._top_k      = top_k
        self._lock       = _RWLock()
        self._executor   = ThreadPoolExecutor(max_workers=workers)

        self._index: faiss.Index
        self._ids:   list[str]   # parallel to FAISS internal order
        self._load()

    # ── internal: load ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                with self._meta_path.open("rb") as fh:
                    self._ids = pickle.load(fh)
                logger.info(
                    "VectorStore loaded: %d vectors from %s",
                    self._index.ntotal, self._index_path,
                )
                return
            except Exception as exc:
                logger.error(
                    "VectorStore load failed (%s); creating fresh index.", exc
                )

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = faiss.IndexFlatL2(DIM)
        self._ids   = []
        logger.info("VectorStore: created fresh IndexFlatL2(dim=%d)", DIM)

    # ── internal: save (must hold write lock) ────────────────────────────────

    def _save_locked(self) -> bool:
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))
            with self._meta_path.open("wb") as fh:
                pickle.dump(self._ids, fh)
            logger.debug("VectorStore saved: %d vectors", self._index.ntotal)
            return True
        except Exception as exc:
            logger.error("VectorStore save failed: %s", exc)
            return False

    # ── internal: executor helper ─────────────────────────────────────────────

    async def _run(self, fn, *args):
        """Run a sync callable in the thread-pool without blocking the loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # ─────────────────────────────────────────────────────────────────────────
    # Public async API
    # ─────────────────────────────────────────────────────────────────────────

    # ── size ──────────────────────────────────────────────────────────────────

    async def get_size(self) -> int:
        """Return number of vectors currently stored."""
        return await self._run(self._get_size_sync)

    def _get_size_sync(self) -> int:
        with self._lock.reading():
            return self._index.ntotal

    # ── user_exists ───────────────────────────────────────────────────────────

    async def user_exists(self, user_id: str) -> bool:
        return await self._run(self._user_exists_sync, user_id)

    def _user_exists_sync(self, user_id: str) -> bool:
        with self._lock.reading():
            return user_id in self._ids

    async def search_one(
            self, embedding: np.ndarray
    ) -> tuple[Optional[str], float]:
        """
        Search for a single embedding.

        Returns
        ───────
        (user_id, similarity)  if a match is found above threshold
        (None,    similarity)  if best match is below threshold
        (None,    0.0)         if index is empty
        """
        results = await self.search_batch(embedding)
        match = results[0]

        if match is None:
            return None, 0.0

        return match.user_id, match.similarity




    # ── search_batch ──────────────────────────────────────────────────────────


    async def search_batch(
        self, embeddings: np.ndarray
    ) -> list[Optional[MatchResult]]:
        """
        Search for N embeddings in one FAISS call.

        Parameters
        ──────────
        embeddings : float32 array shape (N, 512) or (512,) — L2-normalised.

        Returns
        ───────
        list[MatchResult | None]  length N.
        None if no match passes the threshold for that query.
        """
        return await self._run(self._search_batch_sync, embeddings)


    def _search_batch_sync(
            self, embeddings: np.ndarray
    ) -> list[Optional[MatchResult]]:
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis]

        n = embeddings.shape[0]
        results: list[Optional[MatchResult]] = [None] * n

        with self._lock.reading():
            if self._index.ntotal == 0:
                return results

            emb = embeddings.astype(np.float32)
            distances, indices = self._index.search(emb, self._top_k)

            for i in range(n):
                best_dist = distances[i, 0]
                best_idx = indices[i, 0]

                if best_idx < 0 or best_idx >= len(self._ids):
                    continue

                match = MatchResult(self._ids[best_idx], best_dist)

                if best_dist <= self._threshold:
                    # ✅ within threshold → accepted match
                    results[i] = match
                else:
                    # ❌ outside threshold → no match, but log score for debugging
                    logger.debug(
                        "no_match: user_id=%s  dist=%.4f  sim=%.4f  threshold=%.4f",
                        match.user_id, best_dist, match.similarity, self._threshold,
                    )

        return results


    # ── add ───────────────────────────────────────────────────────────────────

    async def add(
        self, user_id: str, embedding: np.ndarray
    ) -> tuple[bool, str]:
        """Add a new user embedding. Returns (False, reason) if already exists."""
        return await self._run(self._add_sync, user_id, embedding)

    def _add_sync(
        self, user_id: str, embedding: np.ndarray
    ) -> tuple[bool, str]:
        if embedding.shape != (DIM,):
            return False, f"Embedding must be shape ({DIM},)"

        with self._lock.writing():
            if user_id in self._ids:
                return False, f"User '{user_id}' already registered"

            vec = embedding.astype(np.float32).reshape(1, -1)
            self._index.add(vec)
            self._ids.append(user_id)
            self._save_locked()

        logger.info(
            "VectorStore: added '%s'  total=%d", user_id, self._index.ntotal
        )
        return True, "OK"

    # ── upsert ────────────────────────────────────────────────────────────────

    async def upsert(
        self, user_id: str, embedding: np.ndarray
    ) -> tuple[bool, str]:
        """Add or replace embedding for user_id."""
        return await self._run(self._upsert_sync, user_id, embedding)

    def _upsert_sync(
        self, user_id: str, embedding: np.ndarray
    ) -> tuple[bool, str]:
        with self._lock.writing():
            if user_id in self._ids:
                self._delete_locked(user_id)

            vec = embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            self._index.add(vec)
            self._ids.append(user_id)
            self._save_locked()

        logger.info(
            "VectorStore: upserted '%s'  total=%d", user_id, self._index.ntotal
        )
        return True, "OK"

    # ── delete ────────────────────────────────────────────────────────────────

    async def delete(self, user_id: str) -> tuple[bool, str]:
        """Remove a user's embedding."""
        return await self._run(self._delete_sync, user_id)

    def _delete_sync(self, user_id: str) -> tuple[bool, str]:
        with self._lock.writing():
            if user_id not in self._ids:
                return False, f"User '{user_id}' not found"
            self._delete_locked(user_id)
            self._save_locked()

        logger.info(
            "VectorStore: deleted '%s'  total=%d", user_id, self._index.ntotal
        )
        return True, "OK"

    def _delete_locked(self, user_id: str) -> None:
        """Rebuild index minus the deleted user (must hold write lock)."""
        idx   = self._ids.index(user_id)
        total = self._index.ntotal

        remaining_vecs: list[np.ndarray] = []
        remaining_ids:  list[str]        = []

        for i in range(total):
            if i == idx:
                continue
            try:
                remaining_vecs.append(self._index.reconstruct(i))
                remaining_ids.append(self._ids[i])
            except Exception as exc:
                logger.warning("reconstruct(%d) failed: %s", i, exc)

        new_index = faiss.IndexFlatL2(DIM)
        if remaining_vecs:
            arr = np.vstack(remaining_vecs).astype(np.float32)
            faiss.normalize_L2(arr)
            new_index.add(arr)

        self._index = new_index
        self._ids   = remaining_ids

    # ── save ──────────────────────────────────────────────────────────────────

    async def save(self) -> bool:
        """Manually flush index + metadata to disk."""
        return await self._run(self._save_sync)

    def _save_sync(self) -> bool:
        with self._lock.writing():
            return self._save_locked()

    # ── close ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the thread-pool. Call once on application shutdown."""
        self._executor.shutdown(wait=True)
        logger.info("VectorStore executor shut down.")

    async def health(self) -> dict:
        """
        Mirror VectorStoreClient.health() interface.
        Returns local store stats instead of an HTTP call.
        """
        size = await self.get_size()
        return {
            "status": "ok",
            "vectors_stored": size,
            "embedding_dim": DIM,
            "threshold": self._threshold,
        }