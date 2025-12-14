from pydantic import BaseModel

class TopKItem(BaseModel):
        class_name: str
        prob: float

class PredictResponse(BaseModel):
        predicted_class: str
        confidence: float
        top3: list[TopKItem]
        model_version: str
        latency_ms: float




