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

from fastapi import FastAPI

app = FastAPI()


@app.post("/predict")
def predict(image):

    return top_k_names, probabilities, predicted, model_ver, latency

@app.get("/health")
def get_health():
    print("Process is running...")

@app.get("/ready")
def get_ready():
    print("Model checkpoint loaded successfully")


