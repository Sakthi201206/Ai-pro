import faiss
import numpy as np
import pandas as pd
from backend.core.embeddings import get_embedding
from backend.config import DATASET_PATH

index = None
texts = []
labels = []
topics = []
sources = []
years = []
strengths = []
sentiments = []


def _load_index() -> None:
    global index, texts, labels, topics, sources, years, strengths, sentiments

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["text"])

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    topics = df["topic"].tolist()
    sources = df["source_type"].tolist()
    years = df["year"].tolist()
    strengths = df["argument_strength"].tolist()
    sentiments = df["sentiment_score"].tolist()

    print("Building FAISS index...")
    embeddings = get_embedding(texts)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)   # Inner-product (cosine on normalized vecs)
    index.add(embeddings.astype("float32"))
    print(f"Index ready — {len(texts):,} documents")


def _ensure_index_loaded() -> None:
    if index is None:
        _load_index()


def retrieve(query: str, k: int = 40) -> list[dict]:
    """Return top-k most relevant documents for the query."""
    _ensure_index_loaded()

    q_emb = get_embedding([query]).astype("float32")
    scores, indices = index.search(q_emb, k)

    results = []
    seen_texts = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        text = texts[idx]
        if text in seen_texts:
            continue  # Skip duplicates
        seen_texts.add(text)
        results.append({
            "text":      text,
            "label":     labels[idx],
            "topic":     topics[idx],
            "source":    sources[idx],
            "year":      int(years[idx]),
            "strength":  float(strengths[idx]),
            "sentiment": float(sentiments[idx]),
            "score":     float(score),
        })
        if len(results) >= 10:  # Limit to 10 unique docs
            break
    return results


def get_available_topics() -> list[str]:
    """Return a sorted list of unique topics from the dataset."""
    _ensure_index_loaded()
    return sorted({t.strip() for t in topics if isinstance(t, str) and t.strip()})
