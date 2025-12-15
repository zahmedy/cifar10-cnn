"""FastAPI service that exposes a CIFAR-10 classifier.

Endpoints:
- GET /health: basic liveness check
- GET /ready: readiness check (model + checkpoint)
- POST /predict: upload an image, receive top-3 predictions and metadata
"""

import io
import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch 

from .model_store import get_model
from .preprocess import preprocess_image
from .schemas import PredictResponse, TopKItem
from .settings import CLASSES, MODEL_VERSION, CHECKPOINT_PATH

logger = logging.getLogger(__name__)


async def lifespan(app: FastAPI):
    """Warm the model once at startup so the first request is fast."""
    model = get_model()
    model.eval()

    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(1, 3, 32, 32, device=device)
        _ = model(dummy)
    logger.info("Model warmed and ready for inference.")
    yield

app = FastAPI(title="CIFAR-10 Image CLassifier", version=MODEL_VERSION, lifespan=lifespan)

@app.get("/health")
def get_health():
    return {"status": "ok"}

@app.get("/ready")
def get_ready():
    if not CHECKPOINT_PATH.exists():
        return {"ready": False, "reason": f"missing checkpoint: {CHECKPOINT_PATH}"}
    try:
        _ = get_model()
        return {"ready": True, "model_version": MODEL_VERSION, "device": str(_.parameters().__next__().device)}
    except Exception as e:
        return {"ready": False, "reason": str(e)}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    
    data = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")
    
    curr_time = time.time()

    processed_img = preprocess_image(pil_image)
    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Failed to load model during predict")
        raise HTTPException(status_code=500, detail="Model not available") from exc
    
    with torch.no_grad():
        logits = model(processed_img)
        probs = torch.softmax(logits ,dim=-1)
    
        top_vals, top_idxs = probs.topk(3, dim=-1)
    
    latency = (time.time() - curr_time) * 1000.0

    pred_idx = int(top_idxs[0, 0].item())
    pred_class = CLASSES[pred_idx]
    pred_conf = float(top_vals[0, 0].item())
        
    top3 = [
        TopKItem(class_name=CLASSES[int(idx.item())], prob=float(val.item()))
        for val, idx in zip(top_vals[0], top_idxs[0])
    ]

    return PredictResponse(
            predicted_class=pred_class, 
            confidence=pred_conf,
            top3=top3,
            model_version=MODEL_VERSION,
            latency_ms=latency
    )
