from dotenv import load_dotenv
import os
import sys

load_dotenv()

HOST = os.getenv("EMB_MODEL_HOST", "0.0.0.0")
PORT = int(os.getenv("EMB_MODEL_PORT", "8004"))
WORKERS = int(os.getenv("EMB_NUM_WORKER", "1"))




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.inference_api:app",
        host=HOST,
        port=PORT,
        # workers > 1 crashes on Windows — use 1 always
        workers=1 if sys.platform == "win32" else WORKERS,
        reload=False,
    )