"""
vector_store_client.py
────────────────────────────────────────────────────────────────────────────
HTTP client that talks to the FAISS Vector Store API (machine-c).
Replaces direct VectorStore usage — all calls go over the network.
"""

from __future__ import annotations

import logging
from typing import Optional
import httpx
import numpy as np

logger = logging.getLogger("vector_store_client")




class MatchResult:
    """Mirrors VectorStore.MatchResult but built from API response JSON."""
    __slots__ = ("user_id", "distance", "similarity")

    def __init__(self, user_id: str, distance: float, similarity: float) -> None:
        self.user_id    = user_id
        self.distance   = distance
        self.similarity = similarity

    def __repr__(self) -> str:
        return f"MatchResult({self.user_id!r}, sim={self.similarity:.3f})"



class VectorStoreClient:
    def __init__(self, base_url: str = "http://10.29.208.81:8005", timeout: float = 5.0):
        self._base   = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url = self._base,
            timeout  = timeout,
            headers  = {"Content-Type": "application/json"},
        )


        logger.info("VectorStoreClient → %s", self._base)  # ← add this

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _post(self, path: str, **json_body) -> dict:
        r = await self._client.post(path, json=json_body)
        r.raise_for_status()
        return r.json()


    async def _get(self, path: str, **params) -> dict:
        r = await self._client.get(path, params=params or None)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parse_match(data: Optional[dict]) -> Optional[MatchResult]:
        if data is None:
            return None
        return MatchResult(
            user_id=data["user_id"],
            distance=data["distance"],
            similarity=data["similarity"],
        )



    # ── read ─────────────────────────────────────────────────────────────────

    # ✅ Fix — match VectorStore exactly
    async def get_size(self) -> int:
        return (await self._get("/stats"))["vectors_stored"]

    async def user_exists(self, user_id: str) -> bool:
        return (await self._get(f"/exists/{user_id}"))["exists"]

    async def search_batch(self, embeddings: np.ndarray,threshold:float=None) -> list[Optional[MatchResult]]:
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis]
        payload = embeddings.astype(np.float32).tolist()
        data    = await self._post("/batch", embeddings=payload)
        return [self._parse_match(r) for r in data["results"]]


    # ── write ─────────────────────────────────────────────────────────────────

    async def add(self, user_id: str, embedding: np.ndarray) -> tuple[bool, str]:
        try:
            await self._post("/register", user_id=user_id,
                             embedding=embedding.astype(np.float32).tolist())
            return True, "OK"
        except httpx.HTTPStatusError as e:
            msg = e.response.json().get("detail", str(e))
            return False, msg

    async def upsert(self, user_id: str, embedding: np.ndarray) -> tuple[bool, str]:
        try:
            await self._post("/upsert", user_id=user_id,
                             embedding=embedding.astype(np.float32).tolist())
            return True, "OK"
        except httpx.HTTPStatusError as e:
            msg = e.response.json().get("detail", str(e))
            return False, msg

    async def delete(self, user_id: str) -> tuple[bool, str]:
        try:
            r = await self._client.delete(f"/delete/{user_id}")
            r.raise_for_status()
            return True, "OK"
        except httpx.HTTPStatusError as e:
            msg = e.response.json().get("detail", str(e))
            return False, msg

    async def search_one(
            self, embedding: np.ndarray
    ) -> tuple[Optional[str], float]:
        """
        Search for a single embedding against the remote vector store.

        Returns
        ───────
        (user_id, similarity)  if match found above threshold
        (None,    similarity)  if best match is below threshold
        (None,    0.0)         if index is empty or no results
        """
        results = await self.search_batch(embedding)

        if not results or results[0] is None:
            return None, 0.0

        match = results[0]
        return match.user_id, match.similarity

    async def save(self) -> bool:
        try:
            await self._post("/save")
            return True
        except Exception as exc:
            logger.error("remote save failed: %s", exc)
            return False

    async def health(self) -> dict:
        return await self._get("/health")

    async def close(self) -> None:
        await self._client.aclose()





