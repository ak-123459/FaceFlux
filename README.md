# 🎯 Distributed Face Recognition Attendance System

A distributed, scalable face recognition system for automated attendance marking. Designed to run across multiple machines or on a single instance — camera nodes detect faces, a background worker sends them for recognition, and attendance is recorded automatically.

---

## 📌 What It Does

1. **Camera Node** captures video, detects and crops faces, saves them locally
2. **Background Worker** continuously picks up cropped faces and sends them to the inference server
3. **Inference Server** converts face images into embeddings (512-number fingerprints) using ONNX models
4. **Vector Store** compares embeddings against registered faces using FAISS
5. **Database** records attendance if a match is found — once per person per day

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MASTER NODE                          │
│              (monitors and manages all workers)             │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┼───────────────┐
       │           │               │
┌──────▼──────┐  ┌─▼────────┐  ┌──▼──────────┐
│ Camera Node │  │Inference │  │  Vector DB  │
│ (Machine 1) │  │  Server  │  │   Server    │
│             │  │(Machine 2│  │ (Machine 3) │
│ • Detect    │  │          │  │             │
│ • Crop      │  │ • ONNX   │  │ • FAISS     │
│ • Save      │  │ • Embed  │  │ • Search    │
│             │  │ • Match  │  │ • Store     │
└──────┬──────┘  └────┬─────┘  └──────┬──────┘
       │              │               │
       └──────────────▼───────────────┘
                      │
               ┌──────▼──────┐
               │  SQL Server  │
               │  (MariaDB)   │
               │ • Attendance │
               │ • Users      │
               └─────────────┘
```

---

## 🖥️ Deployment Modes

### Single Machine
All services run on one machine — good for testing or small deployments.

```
localhost:8004  →  Inference API
localhost:8005  →  Vector Store API
localhost:3306  →  MariaDB
```

### Multi Machine (Distributed)
Each service runs on a dedicated machine for maximum performance.

```
Machine 1  →  Camera + Background Worker
Machine 2  →  Inference API (ONNX model)
Machine 3  →  Vector Store API (FAISS)
Machine 4  →  SQL Database (MariaDB)
Master     →  Monitors all worker nodes
```

---

## 🔄 How Attendance Works

```
Camera sees John
      ↓
Detect + crop face → save as JPEG
      ↓
Background worker picks up JPEG
      ↓
POST /attendance/batch  →  Inference API
      ↓
ONNX model converts face → 512-number embedding
      ↓
FAISS searches vector store for closest match
      ↓
Match found (similarity ≥ threshold)?
      ├── YES → Check if already marked today
      │           ├── Already marked → skip
      │           └── Not marked → INSERT into attendance table ✅
      └── NO  → no_match, skip
```

---

## 🔌 API Endpoints

### Inference API (Port 8004)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/faces/register` | Register a new face |
| `POST` | `/attendance/batch` | Mark attendance for N faces |
| `DELETE` | `/faces/{user_id}` | Remove a registered face |
| `GET` | `/attendance` | Query attendance records |
| `GET` | `/health` | System health check |
| `POST` | `/admin/switch-model` | Hot-swap ONNX model |

### Vector Store API (Port 8005)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Store a new face embedding |
| `POST` | `/upsert` | Add or replace embedding |
| `POST` | `/batch` | Search N embeddings at once |
| `POST` | `/search` | Search single embedding |
| `DELETE` | `/delete/{user_id}` | Remove embedding |
| `GET` | `/exists/{user_id}` | Check if user registered |
| `GET` | `/health` | Store health + size |

---

## ⚙️ Configuration

All settings are controlled via `.env` files.

### Inference API `.env`
```env
# Model
EMB_MODEL_PATH=models/w600k_r50_int8.onnx
EMB_MODEL_TYPE=arcface                    # arcface | vggface2-face

# Vector Store connection
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=8005
VECTOR_STORE_TIMEOUT=5.0

# Server
EMB_MODEL_HOST=0.0.0.0
EMB_MODEL_PORT=8004
EMB_NUM_WORKER=1
THREAD_POOL_SIZE=4
MAX_BATCH_SIZE=64
```

### Vector Store `.env`
```env
FAISS_INDEX_PATH=data/faiss.index
FAISS_META_PATH=data/faiss_meta.pkl

# Threshold: L2 distance for a positive match
# L2=0.63 → cosine≈0.80  ← recommended
# L2=0.50 → cosine≈0.875 ← strict
MATCH_THRESHOLD=0.63

TOP_K=1
VECTOR_DB_HOST=0.0.0.0
VECTOR_DB_PORT=8005
VECTOR_DB_WORKER=4
```

---

## 🧠 Supported Models

| Model | Type | Input Size | Notes |
|-------|------|------------|-------|
| `w600k_r50.onnx` | ArcFace | 112×112 | Default, fast |
| `w600k_r50_int8.onnx` | ArcFace INT8 | 112×112 | Quantized, faster |
| `vggface2_int8.onnx` | VGGFace2 INT8 | 160×160 | Higher accuracy |

All models output a **512-dimensional L2-normalized embedding**.

---

## 📊 Threshold Guide

### Registration Threshold (duplicate check)
```
0.70  →  loose   (easy to register new people)
0.80  →  default (recommended ✅)
0.90  →  strict  (very similar faces blocked)
```

### FAISS Matching Threshold (attendance)
```
L2 value  →  Cosine equivalent  →  Behaviour
  0.40    →      0.920          →  very strict
  0.50    →      0.875          →  strict
  0.63    →      0.800          →  recommended ✅
  0.77    →      0.700          →  loose
  1.00    →      0.500          →  very loose
```

---

## 📝 Attendance Rules

- **One record per person per day** — duplicate scans are silently ignored
- **PRESENT** if first scan before 09:15
- **LATE** if first scan after 09:15
- Protected by two layers: Python `already_marked()` check + SQL `INSERT IGNORE`

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install fastapi uvicorn onnxruntime opencv-python faiss-cpu numpy httpx python-dotenv
```

### 2. Start Vector Store API
```bash
cd vector_store_service
uvicorn main:app --host 0.0.0.0 --port 8005
```

### 3. Start Inference API
```bash
cd inference_service
uvicorn inference_api:app --host 0.0.0.0 --port 8004
```

### 4. Register a face
```bash
curl -X POST http://localhost:8004/faces/register \
  -F "files=@john_face.jpg" \
  -F "name=John Doe" \
  -F "department=Engineering" \
  -F "threshold=0.80"
```

### 5. Mark attendance
```bash
curl -X POST http://localhost:8004/attendance/batch \
  -F "files=@captured_face.jpg" \
  -F "camera_id=CAM_01"
```

---

## 📁 Project Structure

```
├── inference_service/
│   ├── inference_api.py          # FastAPI — main entry point
│   ├── pipelines/
│   │   ├── batch_pipeline.py     # Attendance + embedding pipeline
│   │   └── onnx_embed_engine.py  # ONNX model wrapper (singleton)
│   ├── services/
│   │   └── vector_store_client.py # HTTP client → Vector Store API
│   └── database/
│       └── database.py           # MariaDB connection + attendance writer
│
├── vector_store_service/
│   ├── main.py                   # FastAPI — FAISS API
│   └── database/
│       └── vector_store_db.py    # Thread-safe FAISS wrapper
│
├── camera_node/
│   ├── capture.py                # Face detection + crop + save
│   └── worker.py                 # Background worker → sends to inference API
│
└── master/
    └── monitor.py                # Watches all worker nodes
```

---

## 🔍 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `no_match` returned | FAISS threshold too strict | Set `MATCH_THRESHOLD=0.63` |
| Same person registers twice | Registration threshold too low | Set `threshold=0.80` |
| `similarity: 0.0` in response | Match is `None` (below threshold) | Check threshold + verify correct image sent |
| `INVALID_ARGUMENT 160 Expected 112` | Wrong `model_type` for model | Set `EMB_MODEL_TYPE=arcface` for ArcFace models |
| Attendance not marking | Already marked today | Expected — one record per day per person |