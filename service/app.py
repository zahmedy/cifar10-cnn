"""
App.py an API endpoint for Imgae Classfification built with FastAPI

1) POST /predict

Input: an image (simplest: upload a file)

Output:
top_k class names
probabilities
predicted class
model version
latency_ms

2) GET /health and GET /ready

/health: process is running
/ready: model checkpoint loaded successfully
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from .model_store import get_model
from .preprocess import preprocess_image
from .schemas import PredictResponse, TopKItem
from .settings import CLASSES, MODEL_VERSION, CHECKPOINT_PATH
import torch 
import time
import io


async def lifespan(app: FastAPI):
    """Warm the model once at startup so the first request is fast."""
    model = get_model()
    model.eval()
    
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(1, 3, 32, 32, device=device)
        _ = model(dummy)
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
    model = get_model()
    
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

