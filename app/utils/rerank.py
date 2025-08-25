import os
from typing import List, Tuple


def _approx_trim(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    # crude approx: 1 token ~= 0.75 words
    words = text.split()
    max_words = int(max_tokens / 0.75)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _tfidf_scores(query: str, texts: List[str]) -> List[Tuple[int, float]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        # fallback: simple length heuristic
        return [(i, len(t)) for i, t in enumerate(texts)]

    vect = TfidfVectorizer(stop_words="english", max_features=8192)
    matrix = vect.fit_transform([query] + texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    return list(enumerate(sims))


def _cross_encoder_scores(query: str, texts: List[str], model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        ce = CrossEncoder(model_name)
        pairs = [(query, t) for t in texts]
        scores = ce.predict(pairs)
        return list(enumerate(scores))
    except Exception:
        return None


def rerank_texts(query: str, texts: List[str], top_k: int = 3, max_tokens: int = 300) -> List[str]:
    if not texts:
        return []

    # Optional local cross-encoder
    use_ce = os.getenv("USE_LOCAL_RERANKER", "0") == "1"
    ce_model = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = None
    if use_ce:
        scores = _cross_encoder_scores(query, texts, ce_model)

    # Fallback to TF-IDF if cross-encoder unavailable
    if scores is None:
        scores = _tfidf_scores(query, texts)

    # Sort by score desc
    scores.sort(key=lambda x: x[1], reverse=True)
    indices = [i for i, _ in scores[: max(1, top_k)]]
    selected = [texts[i] for i in indices]

    # Trim each selected chunk to max_tokens/selected count
    per_chunk = max(1, int(max_tokens / max(1, len(selected))))
    return [_approx_trim(t, per_chunk) for t in selected]

