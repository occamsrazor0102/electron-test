"""Semantic Flow analysis — inter-sentence embedding similarity variance.

LLMs produce uniformly smooth transitions between sentences (low variance
in consecutive cosine similarities). Human writing jumps between ideas
erratically, producing high variance.

This signal is resistant to simple paraphrasing attacks because it measures
structural rhythm rather than surface-level token patterns.
"""

import statistics

import numpy as np

from llm_detector.compat import HAS_SEMANTIC, get_semantic_models
from llm_detector.text_utils import get_sentences

_COSINE_EPS = 1e-12

try:
    from sklearn.metrics.pairwise import cosine_similarity

    def _cosine(a, b):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < _COSINE_EPS:
            return 0.0
        return float(cosine_similarity(
            a.reshape(1, -1),
            b.reshape(1, -1),
        )[0][0])
except ImportError:
    def _cosine(a, b):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if abs(denom) < _COSINE_EPS:
            return 0.0
        return float(np.dot(a, b) / denom)

_FLOW_EMPTY = {
    'flow_similarities': [],
    'flow_mean': 0.0,
    'flow_variance': 0.0,
    'flow_std': 0.0,
    'n_sentences': 0,
    'determination': None,
    'confidence': 0.0,
    'reason': '',
}


def run_semantic_flow(text, min_sentences=5):
    """Compute inter-sentence embedding similarity variance.

    Encodes each sentence with the shared all-MiniLM-L6-v2 embedder,
    then measures the variance of cosine similarities between consecutive
    sentence pairs.

    Args:
        text: Input text.
        min_sentences: Minimum sentences required for analysis.

    Returns dict with flow_variance (key signal), determination, confidence.
    """
    if not HAS_SEMANTIC:
        return {**_FLOW_EMPTY, 'reason': 'Semantic flow unavailable (sentence-transformers not installed)'}

    sentences = get_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < min_sentences:
        return {**_FLOW_EMPTY, 'n_sentences': len(sentences),
                'reason': f'Semantic flow: too few sentences ({len(sentences)})'}

    embedder, _, _ = get_semantic_models()

    # Encode all sentences in one batch for efficiency
    embeddings = embedder.encode(sentences)

    # Compute consecutive cosine similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(_cosine(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)

    if len(similarities) < 2:
        return {**_FLOW_EMPTY, 'n_sentences': len(sentences),
                'reason': 'Semantic flow: insufficient sentence pairs'}

    flow_mean = statistics.mean(similarities)
    flow_variance = statistics.variance(similarities)
    flow_std = statistics.stdev(similarities)

    # Determination: low variance + reasonably high mean = AI smoothness
    if flow_variance < 0.008 and flow_mean > 0.40:
        det = 'AMBER'
        conf = min(0.55, 0.30 + (0.008 - flow_variance) * 30.0)
        reason = (f"Very smooth semantic flow (var={flow_variance:.4f}, "
                  f"mean={flow_mean:.3f}): uniformly coherent transitions")
    elif flow_variance < 0.015 and flow_mean > 0.35:
        det = 'YELLOW'
        conf = min(0.30, 0.15 + (0.015 - flow_variance) * 15.0)
        reason = (f"Smooth semantic flow (var={flow_variance:.4f}, "
                  f"mean={flow_mean:.3f}): low transition variance")
    else:
        det = None
        conf = 0.0
        reason = (f"Normal semantic flow (var={flow_variance:.4f}, "
                  f"mean={flow_mean:.3f})")

    return {
        'flow_similarities': [round(s, 4) for s in similarities],
        'flow_mean': round(flow_mean, 4),
        'flow_variance': round(flow_variance, 6),
        'flow_std': round(flow_std, 4),
        'n_sentences': len(sentences),
        'determination': det,
        'confidence': round(conf, 4),
        'reason': reason,
    }
