"""Semantic Resonance -- embedding proximity to AI/human archetype centroids.

Ref: Mitchell et al. (2023) "DetectGPT" -- semantic density as AI signal.
"""

from llm_detector.compat import HAS_SEMANTIC, get_semantic_models

if HAS_SEMANTIC:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity


def run_semantic_resonance(text):
    """Measure semantic similarity to AI vs human archetypes.

    Returns dict with semantic scores, delta, determination, and confidence.
    """
    if not HAS_SEMANTIC:
        return {
            'semantic_ai_score': 0.0,
            'semantic_human_score': 0.0,
            'semantic_delta': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Semantic layer unavailable (sentence-transformers not installed)',
        }

    words = text.split()
    if len(words) < 30:
        return {
            'semantic_ai_score': 0.0,
            'semantic_human_score': 0.0,
            'semantic_delta': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Semantic layer: text too short',
        }

    chunk_size = 200
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)

    if not chunks:
        chunks = [text]

    embedder, ai_centroids, human_centroids = get_semantic_models()

    vecs = embedder.encode(chunks)

    ai_sims = _cosine_similarity(vecs, ai_centroids)
    max_ai_sim = float(ai_sims.max())
    mean_ai_sim = float(ai_sims.max(axis=1).mean())

    human_sims = _cosine_similarity(vecs, human_centroids)
    max_human_sim = float(human_sims.max())
    mean_human_sim = float(human_sims.max(axis=1).mean())

    semantic_delta = mean_ai_sim - mean_human_sim

    if mean_ai_sim >= 0.65 and semantic_delta >= 0.15:
        det = 'AMBER'
        conf = min(0.60, mean_ai_sim)
        reason = f"Semantic resonance: AI={mean_ai_sim:.2f}, delta={semantic_delta:.2f}"
    elif mean_ai_sim >= 0.50 and semantic_delta >= 0.08:
        det = 'YELLOW'
        conf = min(0.35, mean_ai_sim)
        reason = f"Semantic resonance: AI={mean_ai_sim:.2f}, delta={semantic_delta:.2f}"
    else:
        det = None
        conf = 0.0
        reason = 'Semantic resonance: below threshold'

    return {
        'semantic_ai_score': round(max_ai_sim, 4),
        'semantic_human_score': round(max_human_sim, 4),
        'semantic_ai_mean': round(mean_ai_sim, 4),
        'semantic_human_mean': round(mean_human_sim, 4),
        'semantic_delta': round(semantic_delta, 4),
        'determination': det,
        'confidence': conf,
        'reason': reason,
    }
