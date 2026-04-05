from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings for a list of texts."""
    model = _load_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings
