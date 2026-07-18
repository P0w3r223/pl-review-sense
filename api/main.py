"""FastAPI service: classify Polish review sentiment with the TF-IDF baseline.

The baseline is fast and light enough to serve directly; HerBERT is not served here (it is a
Colab-trained artifact). The model is loaded once at startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pl_review_sense import baseline, config
from pl_review_sense.data import label_name

_state: dict = {"model": None}


class ReviewIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000, examples=["Świetny produkt, gorąco polecam!"])


class SentimentOut(BaseModel):
    label: str
    label_index: int
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["model"] = baseline.load()
    except FileNotFoundError:
        _state["model"] = None  # /health reports it; /predict returns 503
    yield


app = FastAPI(
    title="pl-review-sense",
    description="Polish review sentiment — TF-IDF + logistic-regression baseline",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=SentimentOut)
def predict(review: ReviewIn) -> SentimentOut:
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="model not loaded — run `python -m pl_review_sense.baseline_train` first",
        )
    proba = baseline.predict_proba(model, [review.text])[0]
    index = max(range(len(proba)), key=lambda i: proba[i])
    return SentimentOut(
        label=label_name(index),
        label_index=index,
        probabilities={config.LABEL_NAMES[i]: round(proba[i], 4) for i in range(len(proba))},
    )
