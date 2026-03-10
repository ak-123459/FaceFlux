from __future__ import annotations
from pydantic import BaseModel
from typing import Annotated, Optional
import logging
import os
import time
from pydantic import BaseModel, Field, field_validator
from database.vector_store_db import  DIM
from dataclasses import dataclass, field



# ─────────────────────────────────────────────────────────────────────────────
# Schemas  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class FrameOut(BaseModel):
    frame_index:    int
    filename:       Optional[str]
    matched:        bool
    user_id:        Optional[str]
    similarity:     float
    f: bool
    error:          Optional[str]


class BatchOut(BaseModel):
    total:          int
    matched:        int
    attendance_new: int
    errors:         int
    elapsed_ms:     float
    frames:         list[FrameOut]

class RegisterOut(BaseModel):
    success: bool
    user_id: str
    message: str



class DeleteOut(BaseModel):
    success: bool
    message: str

class HealthOut(BaseModel):
    status:           str
    registered_faces: int
    db_ok:            bool
    vector_store_url: str
    vector_store_ok:  bool




class EmbeddingIn(BaseModel):
    """Single embedding payload — sent by your inference API."""
    user_id: str = Field(..., description="Unique user identifier")
    embedding: list[float] = Field(..., description=f"Float32 vector of length {DIM}")

    @field_validator("embedding")
    @classmethod
    def check_dim(cls, v: list[float]) -> list[float]:
        if len(v) != DIM:
            raise ValueError(f"embedding must have exactly {DIM} values, got {len(v)}")
        return v

class QueryIn(BaseModel):
    """Search / match payload — just the embedding, no user_id needed."""
    embedding: list[float] = Field(..., description=f"Float32 query vector of length {DIM}")

    @field_validator("embedding")
    @classmethod
    def check_dim(cls, v: list[float]) -> list[float]:
        if len(v) != DIM:
            raise ValueError(f"embedding must have exactly {DIM} values, got {len(v)}")
        return v

class VerifyIn(BaseModel):
    """1:1 verification — check if embedding matches a specific user."""
    user_id: str
    embedding: list[float] = Field(..., description=f"Float32 query vector of length {DIM}")

    @field_validator("embedding")
    @classmethod
    def check_dim(cls, v: list[float]) -> list[float]:
        if len(v) != DIM:
            raise ValueError(f"embedding must have exactly {DIM} values, got {len(v)}")
        return v

class BatchQueryIn(BaseModel):
    """Batch search — N embeddings in one call."""
    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")

    @field_validator("embeddings")
    @classmethod
    def check_dims(cls, v: list[list[float]]) -> list[list[float]]:
        for i, emb in enumerate(v):
            if len(emb) != DIM:
                raise ValueError(f"embeddings[{i}] must have {DIM} values, got {len(emb)}")
        return v







@dataclass
class FrameResult:
    frame_index:    int
    filename:       Optional[str]  = None
    matched:        bool           = False
    user_id:        Optional[str]  = None
    similarity:     float          = 0.0
    attendance_new: bool           = False
    error:          Optional[str]  = None


@dataclass
class BatchResult:
    total:          int   = 0
    matched:        int   = 0
    attendance_new: int   = 0
    errors:         int   = 0
    elapsed_ms:     float = 0.0
    frames:         list[FrameResult] = field(default_factory=list)




