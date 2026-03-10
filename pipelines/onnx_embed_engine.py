"""
onnx_embed_engine.py
────────────────────────────────────────────────────────────────────────────
Singleton ONNX embedding engine.

Input  : pre-cropped, pre-aligned BGR face image (any size, resized to 112x112)
Output : L2-normalised float32 embedding of shape (512,)

No detection. No landmark alignment. Caller guarantees the image is a
face crop already.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)




def _build_session(model_path: str, providers: list[str]) -> ort.InferenceSession:
    available = ort.get_available_providers()
    selected  = [p for p in providers if p in available] or ["CPUExecutionProvider"]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1

    session = ort.InferenceSession(model_path, sess_options=opts, providers=selected)
    logger.info("Loaded %s on %s", Path(model_path).name, session.get_providers())
    return session


class OnnxEmbedEngine:
    _instances: dict[str, "OnnxEmbedEngine"] = {}  # keyed by model_path
    _lock = threading.Lock()

    @classmethod
    def instance(
        cls,
        model_path: str = "models/w600k_r50.onnx",
        providers: Optional[list[str]] = None,
        model_type: str = "arcface",          # ← NEW: "arcface" | "vggface2-face"
    ) -> "OnnxEmbedEngine":
        if model_path not in cls._instances:
            with cls._lock:
                if model_path not in cls._instances:
                    cls._instances[model_path] = cls(model_path, providers, model_type)
        return cls._instances[model_path]

    def __init__(self, model_path: str, providers: Optional[list[str]], model_type: str = "arcface") -> None:
        self._providers  = providers or ["OpenVINOExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model_type = model_type                    # ← NEW
        self._session    = _build_session(model_path, self._providers)
        self._in_name    = self._session.get_inputs()[0].name
        self._out_name   = self._session.get_outputs()[0].name
        logger.info("OnnxEmbedEngine ready. model_type=%s", model_type)

    def _preprocess(self, crop: np.ndarray, resized: np.ndarray) -> np.ndarray:
        """
        Returns CHW float32 tensor for one crop.
        arcface : BGR 112x112  (x - 127.5) / 128.0
        vggface2-face: RGB 160x160   x/255 → (x - 0.5) / 0.5
        """
        if self._model_type == "vggface2-face":
            resized = cv2.resize(crop, (160, 160))
            arr     = resized[:, :, ::-1]                        # BGR → RGB
            arr     = arr.astype(np.float32) / 255.0
            arr     = (arr - 0.5) / 0.5                          # [-1, 1]
        else:  # arcface / buffalo
            resized = cv2.resize(crop, (112, 112))
            arr     = resized[:, :, ::-1]                        # BGR → RGB
            arr     = (arr.astype(np.float32) - 127.5) / 128.0  # [-1, 1]

        return arr.transpose(2, 0, 1)  # HWC → CHW

    def embed_batch(self, bgr_crops: list[np.ndarray]) -> np.ndarray:
        if not bgr_crops:
            return np.empty((0, 512), dtype=np.float32)

        n    = len(bgr_crops)
        size = 160 if self._model_type == "vggface2-face" else 112       # ← NEW
        batch = np.empty((n, 3, size, size), dtype=np.float32)

        for i, crop in enumerate(bgr_crops):
            batch[i] = self._preprocess(crop, crop)                 # ← NEW

        try:
            out = self._session.run([self._out_name], {self._in_name: batch})[0]
        except Exception:
            logger.warning("Batch inference failed, falling back to sequential.")
            out = np.vstack([
                self._session.run([self._out_name], {self._in_name: batch[i:i+1]})[0]
                for i in range(n)
            ])

        return out.astype(np.float32)