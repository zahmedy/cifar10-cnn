

class PredictResponse:
    def __init__(self, predicted_label: str,
                 top_k: list[{"label": str, "prob": float}],
                 model_version: str,
                 latency_ms: float) -> None:
        pass