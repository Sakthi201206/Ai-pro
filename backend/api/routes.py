from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.debate_service import generate_debate
import pandas as pd
from backend.config import DATASET_PATH

router = APIRouter()


class DebateRequest(BaseModel):
    topic: str


class StatsResponse(BaseModel):
    total: int
    pro: int
    con: int
    neutral: int
    topics: list[str]
    sources: list[str]
    years: list[int]


@router.post("/debate")
def debate(req: DebateRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    try:
        return generate_debate(req.topic.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return dataset statistics for the analytics dashboard."""
    df = pd.read_csv(DATASET_PATH)
    return {
        "total":   len(df),
        "pro":     int((df["label"] == "pro").sum()),
        "con":     int((df["label"] == "con").sum()),
        "neutral": int((df["label"] == "neutral").sum()),
        "topics":  sorted(df["topic"].dropna().unique().tolist()),
        "sources": sorted(df["source_type"].dropna().unique().tolist()),
        "years":   sorted(df["year"].dropna().unique().astype(int).tolist()),
    }


@router.get("/analytics")
def get_analytics():
    """Return full analytics breakdown for charts."""
    df = pd.read_csv(DATASET_PATH)
    label_counts = df["label"].value_counts().to_dict()
    source_counts = df["source_type"].value_counts().to_dict()
    year_label = (
        df.groupby(["year", "label"])
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    topic_counts = df["topic"].value_counts().to_dict()
    avg_strength = float(df["argument_strength"].mean())
    return {
        "label_distribution":  label_counts,
        "source_distribution": source_counts,
        "year_label_trend":    year_label,
        "topic_distribution":  topic_counts,
        "avg_argument_strength": round(avg_strength, 3),
    }


@router.get("/topics")
def get_topics():
    """Return the available debate topics for frontend suggestions."""
    from backend.core.retriever import get_available_topics
    return {"topics": get_available_topics()}


@router.get("/health")
def health():
    return {"status": "ok"}
