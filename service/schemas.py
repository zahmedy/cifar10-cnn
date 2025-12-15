from pydantic import BaseModel
from typing import List


class TopKItem(BaseModel):
    class_name: str
    prob: float


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    top3: List[TopKItem]
    model_version: str
    latency_ms: float
