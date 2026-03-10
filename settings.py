"""
settings.py
────────────────────────────────────────────────────────────────────────────
Single config file for the entire PRISM distributed pipeline.

Each machine imports only what it needs:

  Machine 1 (Capture):
    from config.settings import CaptureSettings

  Machine 2 (Inference):
    from config.settings import InferenceSettings

  Machine 3 (Vector Store):
    from config.settings import VectorStoreSettings

  Machine 4 (SQL):
    from config.settings import SQLSettings

Or auto-detect based on PRISM_ROLE env var:
    from config.settings import settings
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("settings")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _int(key: str, default: int) -> int:
    try:    return int(os.getenv(key, str(default)))
    except: return default

def _float(key: str, default: float) -> float:
    try:    return float(os.getenv(key, str(default)))
    except: return default

def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# Machine 1 — Capture Service
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaptureSettings:
    # Where inference API lives
    inference_host:  str = field(default_factory=lambda: _env("INFERENCE_HOST", "192.168.1.20"))
    inference_port:  int = field(default_factory=lambda: _int("INFERENCE_PORT", 8004))

    # Camera IDs (comma-separated in env)
    camera_ids: list[str] = field(default_factory=lambda: [
        c.strip() for c in _env("CAMERA_IDS", "cam_01").split(",") if c.strip()
    ])

    fps_cap:        int   = field(default_factory=lambda: _int("CAPTURE_FPS_CAP",     5))
    batch_size:     int   = field(default_factory=lambda: _int("CAPTURE_BATCH_SIZE",  8))
    jpeg_quality:   int   = field(default_factory=lambda: _int("CAPTURE_JPEG_QUALITY", 85))

    @property
    def inference_url(self) -> str:
        return f"http://{self.inference_host}:{self.inference_port}"

    def camera_source(self, camera_id: str) -> str:
        """Get RTSP source for a camera_id from env."""
        return _env(f"CAMERA_{camera_id}_SOURCE", "0")

    def log(self):
        logger.info("CaptureSettings:")
        logger.info("  inference_url = %s", self.inference_url)
        logger.info("  cameras       = %s", self.camera_ids)
        logger.info("  fps_cap       = %d", self.fps_cap)
        logger.info("  batch_size    = %d", self.batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Machine 2 — Inference API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceSettings:
    # This service
    host:           str   = field(default_factory=lambda: _env("EMB_MODEL_HOST",  "0.0.0.0"))
    port:           int   = field(default_factory=lambda: _int("EMB_MODEL_PORT",  8004))
    workers:        int   = field(default_factory=lambda: _int("EMB_NUM_WORKER",  1))
    thread_workers: int   = field(default_factory=lambda: _int("THREAD_POOL_SIZE", 4))

    # ONNX model
    model_path:     str   = field(default_factory=lambda: _env("EMB_MODEL_PATH",  "models/w600k_r50_int8.onnx"))

    # Request limits
    max_batch:      int   = field(default_factory=lambda: _int("MAX_BATCH_SIZE",   64))
    camera_id:      str   = field(default_factory=lambda: _env("CAMERA_ID",        ""))

    # ── Vector Store (Machine 3) ──────────────────────────────────────────
    vector_db_host:     str   = field(default_factory=lambda: _env("VECTOR_DB_HOST",         "192.168.1.30"))
    vector_db_port:     int   = field(default_factory=lambda: _int("VECTOR_DB_PORT",         8005))
    vector_db_timeout:  float = field(default_factory=lambda: _float("VECTOR_STORE_TIMEOUT", 5.0))

    # ── SQL (Machine 4) ───────────────────────────────────────────────────
    db_host:        str   = field(default_factory=lambda: _env("DB_HOST",      "192.168.1.40"))
    db_port:        int   = field(default_factory=lambda: _int("DB_PORT",      3306))
    db_user:        str   = field(default_factory=lambda: _env("DB_USER",      "frs_user"))
    db_password:    str   = field(default_factory=lambda: _env("DB_PASSWORD",  ""))
    db_name:        str   = field(default_factory=lambda: _env("DB_NAME",      "attendance_db"))
    db_pool_size:   int   = field(default_factory=lambda: _int("DB_POOL_SIZE", 5))

    @property
    def vector_store_url(self) -> str:
        return f"http://{self.vector_db_host}:{self.vector_db_port}"

    @property
    def safe_workers(self) -> int:
        """Always 1 on Windows — multiprocessing not supported."""
        return 1 if sys.platform == "win32" else self.workers

    def validate(self):
        errors = []
        if not Path(self.model_path).exists():
            errors.append(f"ONNX model not found: {self.model_path}")
        if not self.db_password:
            errors.append("DB_PASSWORD is empty — set it in .env")
        if errors:
            for e in errors:
                logger.error("Config error: %s", e)
            raise ValueError(f"InferenceSettings invalid: {errors}")

    def log(self):
        logger.info("InferenceSettings:")
        logger.info("  host            = %s:%d", self.host, self.port)
        logger.info("  workers         = %d (safe=%d)", self.workers, self.safe_workers)
        logger.info("  model_path      = %s", self.model_path)
        logger.info("  vector_store    = %s", self.vector_store_url)
        logger.info("  db_host         = %s:%d/%s", self.db_host, self.db_port, self.db_name)
        logger.info("  max_batch       = %d", self.max_batch)
        logger.info("  thread_workers  = %d", self.thread_workers)


# ─────────────────────────────────────────────────────────────────────────────
# Machine 3 — Vector Store API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VectorStoreSettings:
    # This service
    host:           str   = field(default_factory=lambda: _env("VECTOR_DB_HOST",  "0.0.0.0"))
    port:           int   = field(default_factory=lambda: _int("VECTOR_DB_PORT",  8005))
    workers:        int   = field(default_factory=lambda: _int("VECTOR_DB_WORKER", 1))
    thread_workers: int   = field(default_factory=lambda: _int("FAISS_THREAD_WORKERS", 4))

    # FAISS
    index_path:     str   = field(default_factory=lambda: _env("FAISS_INDEX_PATH", "data/faiss.index"))
    meta_path:      str   = field(default_factory=lambda: _env("FAISS_META_PATH",  "data/faiss_meta.pkl"))
    threshold:      float = field(default_factory=lambda: _float("MATCH_THRESHOLD", 0.40))
    top_k:          int   = field(default_factory=lambda: _int("TOP_K",             1))

    # Read-only replica mode
    read_only:      bool  = field(default_factory=lambda: _bool("VECTOR_STORE_READ_ONLY", False))

    @property
    def safe_workers(self) -> int:
        return 1 if sys.platform == "win32" else self.workers

    def validate(self):
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.meta_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self):
        logger.info("VectorStoreSettings:")
        logger.info("  host        = %s:%d", self.host, self.port)
        logger.info("  workers     = %d (safe=%d)", self.workers, self.safe_workers)
        logger.info("  index_path  = %s", self.index_path)
        logger.info("  threshold   = %.2f", self.threshold)
        logger.info("  top_k       = %d", self.top_k)
        logger.info("  read_only   = %s", self.read_only)


# ─────────────────────────────────────────────────────────────────────────────
# Machine 4 — SQL Store (config only — MariaDB runs natively)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SQLSettings:
    host:       str = field(default_factory=lambda: _env("DB_HOST",      "0.0.0.0"))
    port:       int = field(default_factory=lambda: _int("DB_PORT",      3306))
    user:       str = field(default_factory=lambda: _env("DB_USER",      "frs_user"))
    password:   str = field(default_factory=lambda: _env("DB_PASSWORD",  ""))
    database:   str = field(default_factory=lambda: _env("DB_NAME",      "attendance_db"))
    pool_size:  int = field(default_factory=lambda: _int("DB_POOL_SIZE", 5))

    def validate(self):
        if not self.password:
            raise ValueError("DB_PASSWORD is required — set it in .env")

    def log(self):
        logger.info("SQLSettings:")
        logger.info("  host      = %s:%d", self.host, self.port)
        logger.info("  database  = %s", self.database)
        logger.info("  user      = %s", self.user)
        logger.info("  pool_size = %d", self.pool_size)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect machine role — used when importing `settings`
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_MAP = {
    "capture":      CaptureSettings,
    "inference":    InferenceSettings,
    "vector_store": VectorStoreSettings,
    "sql":          SQLSettings,
}

def get_settings(role: str = None):
    """
    Return the right settings object for this machine.

    Usage:
        # Explicit
        from config.settings import get_settings
        cfg = get_settings("inference")

        # From env var PRISM_ROLE=inference
        from config.settings import get_settings
        cfg = get_settings()

        # Direct import
        from config.settings import InferenceSettings
        cfg = InferenceSettings()
    """
    role = role or _env("PRISM_ROLE", "inference")
    cls  = _ROLE_MAP.get(role.lower())
    if cls is None:
        raise ValueError(
            f"Unknown PRISM_ROLE='{role}'. "
            f"Valid: {list(_ROLE_MAP.keys())}"
        )
    instance = cls()
    if hasattr(instance, "validate"):
        instance.validate()
    if hasattr(instance, "log"):
        instance.log()
    return instance


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run this on any machine to verify config
# python -m config.settings
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    role = _env("PRISM_ROLE", "")
    if not role:
        print("\nSet PRISM_ROLE to one of: capture, inference, vector_store, sql")
        print("Example:  PRISM_ROLE=inference python -m config.settings\n")
        print("Showing all settings:\n")
        for r in _ROLE_MAP:
            print(f"── {r} ──")
            try:
                s = get_settings(r)
            except Exception as e:
                print(f"  ERROR: {e}")
        sys.exit(0)

    cfg = get_settings(role)
    print(f"\n✅ Config for role '{role}' loaded successfully.\n")