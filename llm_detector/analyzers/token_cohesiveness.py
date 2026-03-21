"""Token Cohesiveness analysis (TOCSIN).

Measures semantic stability under random word deletion.
High cohesiveness (large semantic shift from deletion) = AI signal.

Ref: Ma & Wang (EMNLP 2024), "Zero-Shot Detection of LLM-Generated Text
     using Token Cohesiveness"
"""

import random
import statistics

from llm_detector.compat import HAS_SEMANTIC, get_semantic_models


def run_token_cohesiveness(text, n_copies=10, deletion_rate=0.015, seed=42):
    """Compute token cohesiveness via random word deletion.

    Args:
        text: Input text.
        n_copies: Number of deletion rounds.
        deletion_rate: Fraction of words to delete per round.
        seed: Random seed for reproducibility.

    Returns dict with cohesiveness score, determination, confidence.
    """
    if not HAS_SEMANTIC:
        return {
            'cohesiveness': 0.0,
            'cohesiveness_std': 0.0,
            'n_rounds': 0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Token cohesiveness unavailable (sentence-transformers not installed)',
        }

    words = text.split()
    if len(words) < 60:
        return {
            'cohesiveness': 0.0,
            'cohesiveness_std': 0.0,
            'n_rounds': 0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Token cohesiveness: text too short',
        }

    embedder, _, _ = get_semantic_models()
    from sklearn.metrics.pairwise import cosine_similarity

    orig_embedding = embedder.encode([text])

    rng = random.Random(seed)
    n_delete = max(1, int(len(words) * deletion_rate))
    distances = []

    for _ in range(n_copies):
        indices = list(range(len(words)))
        delete_indices = set(rng.sample(indices, min(n_delete, len(indices) - 1)))
        reduced = ' '.join(w for i, w in enumerate(words) if i not in delete_indices)

        if len(reduced.split()) < 10:
            continue

        reduced_embedding = embedder.encode([reduced])
        sim = cosine_similarity(orig_embedding, reduced_embedding)[0][0]
        distance = float(1.0 - sim)
        distances.append(distance)

    if not distances:
        return {
            'cohesiveness': 0.0,
            'cohesiveness_std': 0.0,
            'n_rounds': 0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Token cohesiveness: all deletion rounds failed',
        }

    cohesiveness = statistics.mean(distances)
    cohesiveness_std = statistics.stdev(distances) if len(distances) >= 2 else 0.0

    if cohesiveness >= 0.020 and cohesiveness_std < 0.010:
        det = 'AMBER'
        conf = min(0.60, 0.30 + cohesiveness * 10.0)
        reason = f"High token cohesiveness ({cohesiveness:.4f}): fragile to word deletion"
    elif cohesiveness >= 0.012:
        det = 'YELLOW'
        conf = min(0.35, 0.15 + cohesiveness * 8.0)
        reason = f"Elevated token cohesiveness ({cohesiveness:.4f})"
    else:
        det = None
        conf = 0.0
        reason = f"Normal token cohesiveness ({cohesiveness:.4f})"

    return {
        'cohesiveness': round(cohesiveness, 6),
        'cohesiveness_std': round(cohesiveness_std, 6),
        'n_rounds': len(distances),
        'determination': det,
        'confidence': round(conf, 4),
        'reason': reason,
    }
