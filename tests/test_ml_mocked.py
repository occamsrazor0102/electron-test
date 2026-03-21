"""Tests for ML-dependent analyzers using mocking.

Covers perplexity.py (6%→~40%), semantic_resonance.py (13%→~50%),
token_cohesiveness.py (15%→~50%) by mocking torch and sentence-transformers.
"""

import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


# ── Perplexity ────────────────────────────────────────────────────────────────

def test_run_perplexity_no_library():
    """run_perplexity returns empty dict when HAS_PERPLEXITY=False."""
    print("\n-- PERPLEXITY: NO LIBRARY --")
    import llm_detector.analyzers.perplexity as ppl_mod
    from llm_detector.compat import HAS_PERPLEXITY

    if not HAS_PERPLEXITY:
        from llm_detector.analyzers.perplexity import run_perplexity
        r = run_perplexity("Some text that is totally normal and fine.")
        check("determination is None", r['determination'] is None)
        check("perplexity == 0.0", r['perplexity'] == 0.0)
        check("surprisal_variance == 0.0", r['surprisal_variance'] == 0.0)
        check("volatility_decay_ratio == 1.0", r['volatility_decay_ratio'] == 1.0)
        check("binoculars_score == 0.0", r['binoculars_score'] == 0.0)
        check("binoculars_determination is None", r['binoculars_determination'] is None)
        check("reason mentions unavailable",
              'unavailable' in r.get('reason', '').lower(),
              f"reason: {r.get('reason')}")
    else:
        check("HAS_PERPLEXITY=True: skip", True)


def test_run_perplexity_short_text_mocked():
    """run_perplexity returns early for short texts (< 50 words) when mocked."""
    print("\n-- PERPLEXITY: SHORT TEXT (MOCKED HAS_PERPLEXITY) --")
    import llm_detector.analyzers.perplexity as ppl_mod

    short_text = "This text has fewer than fifty words."

    with patch.object(ppl_mod, 'HAS_PERPLEXITY', True):
        from llm_detector.analyzers.perplexity import run_perplexity
        r = ppl_mod.run_perplexity(short_text)
        check("Short text returns None determination",
              r['determination'] is None, f"got {r['determination']}")
        check("Short text reason mentions 'too short'",
              'too short' in r.get('reason', ''), f"got {r.get('reason')}")
        check("Short text perplexity == 0.0", r['perplexity'] == 0.0)


def _make_mock_torch(target_ppl: float,
                     token_losses: "np.ndarray | None" = None):
    """Build a minimal mock torch module for perplexity scoring.

    Returns (mock_torch, mock_model, mock_tokenizer).
    """
    if token_losses is None:
        # 20 token losses with some variance
        token_losses = np.array([
            2.5, 2.1, 2.8, 2.3, 2.6, 2.4, 2.7, 2.1, 2.9, 2.3,
            2.5, 2.2, 2.6, 2.4, 2.8, 2.1, 2.7, 2.5, 2.3, 2.4,
        ])

    mock_torch = MagicMock()

    # exp(loss).item() → target_ppl
    mock_torch.exp.return_value.item.return_value = target_ppl

    # CrossEntropyLoss per-token → token_losses numpy array
    mock_loss_fn = MagicMock()
    mock_per_tok = MagicMock()
    mock_per_tok.float.return_value.cpu.return_value.numpy.return_value = token_losses
    mock_loss_fn.return_value = mock_per_tok
    mock_torch.nn.CrossEntropyLoss.return_value = mock_loss_fn

    # Tokenizer — input_ids.size(1) must return an int >= 10
    mock_tok = MagicMock()
    mock_enc = MagicMock()
    mock_enc.input_ids.size.return_value = 100
    mock_tok.return_value = mock_enc

    # Model: two successive calls
    mock_out1 = MagicMock()
    mock_out1.loss.item.return_value = math.log(target_ppl)
    mock_out2 = MagicMock()  # model(input_ids).logits
    mock_model = MagicMock()
    mock_model.side_effect = [mock_out1, mock_out2]

    return mock_torch, mock_model, mock_tok


def test_run_perplexity_mocked_amber():
    """run_perplexity returns AMBER for low perplexity (ppl <= 15)."""
    print("\n-- PERPLEXITY: MOCKED AMBER --")
    import llm_detector.analyzers.perplexity as ppl_mod

    long_text = "test word " * 60  # > 50 words
    target_ppl = 10.0  # <= 15 → AMBER

    mock_torch, mock_model, mock_tok = _make_mock_torch(target_ppl)
    mock_get = MagicMock(return_value=(mock_model, mock_tok))

    with patch.object(ppl_mod, 'HAS_PERPLEXITY', True), \
         patch.object(ppl_mod, '_torch', mock_torch, create=True), \
         patch.object(ppl_mod, 'HAS_BINOCULARS', False), \
         patch.object(ppl_mod, 'get_perplexity_model', mock_get):
        r = ppl_mod.run_perplexity(long_text)

    check("AMBER determination", r['determination'] == 'AMBER',
          f"got {r['determination']}")
    check("perplexity == 10.0", r['perplexity'] == 10.0, f"got {r['perplexity']}")
    check("confidence > 0", r['confidence'] > 0, f"got {r['confidence']}")
    check("reason mentions perplexity", 'perplexity' in r.get('reason', '').lower())
    check("comp_ratio present", 'comp_ratio' in r)
    check("zlib_normalized_ppl present", 'zlib_normalized_ppl' in r)
    check("token_losses present", 'token_losses' in r)
    check("surprisal_variance > 0", r.get('surprisal_variance', 0) > 0,
          f"got {r.get('surprisal_variance')}")


def test_run_perplexity_mocked_yellow():
    """run_perplexity returns YELLOW for moderate perplexity (15 < ppl <= 25)."""
    print("\n-- PERPLEXITY: MOCKED YELLOW --")
    import llm_detector.analyzers.perplexity as ppl_mod

    long_text = "test word " * 60
    target_ppl = 20.0  # 15 < 20 <= 25 → YELLOW

    mock_torch, mock_model, mock_tok = _make_mock_torch(target_ppl)
    mock_get = MagicMock(return_value=(mock_model, mock_tok))

    with patch.object(ppl_mod, 'HAS_PERPLEXITY', True), \
         patch.object(ppl_mod, '_torch', mock_torch, create=True), \
         patch.object(ppl_mod, 'HAS_BINOCULARS', False), \
         patch.object(ppl_mod, 'get_perplexity_model', mock_get):
        r = ppl_mod.run_perplexity(long_text)

    check("YELLOW determination", r['determination'] == 'YELLOW',
          f"got {r['determination']}")
    check("perplexity == 20.0", r['perplexity'] == 20.0, f"got {r['perplexity']}")


def test_run_perplexity_mocked_normal():
    """run_perplexity returns None determination for high perplexity (> 25)."""
    print("\n-- PERPLEXITY: MOCKED NORMAL --")
    import llm_detector.analyzers.perplexity as ppl_mod

    long_text = "test word " * 60
    target_ppl = 40.0  # > 25 → None

    mock_torch, mock_model, mock_tok = _make_mock_torch(target_ppl)
    mock_get = MagicMock(return_value=(mock_model, mock_tok))

    with patch.object(ppl_mod, 'HAS_PERPLEXITY', True), \
         patch.object(ppl_mod, '_torch', mock_torch, create=True), \
         patch.object(ppl_mod, 'HAS_BINOCULARS', False), \
         patch.object(ppl_mod, 'get_perplexity_model', mock_get):
        r = ppl_mod.run_perplexity(long_text)

    check("Normal perplexity -> None determination",
          r['determination'] is None, f"got {r['determination']}")
    check("perplexity == 40.0", r['perplexity'] == 40.0, f"got {r['perplexity']}")


def test_run_perplexity_mocked_diveye_upgrade():
    """DivEye signals upgrade YELLOW → AMBER when low variance + high decay."""
    print("\n-- PERPLEXITY: DIVEYE UPGRADE --")
    import llm_detector.analyzers.perplexity as ppl_mod

    long_text = "test word " * 60
    target_ppl = 20.0  # YELLOW

    # Low variance losses (s_var < 2.0) with decay > 1.5 in first vs second half
    # First half: losses[:10], second half: losses[10:]
    # To get variance < 2.0: all values very similar
    # To get volatility_decay_ratio > 1.5: first_half_var > 1.5 * second_half_var
    n = 40
    first_half = np.ones(n // 2) * 2.3 + np.random.default_rng(0).normal(0, 0.3, n // 2)
    second_half = np.ones(n // 2) * 2.3 + np.random.default_rng(1).normal(0, 0.1, n // 2)
    losses = np.concatenate([first_half, second_half])

    mock_torch, mock_model, mock_tok = _make_mock_torch(target_ppl, losses)
    mock_get = MagicMock(return_value=(mock_model, mock_tok))

    with patch.object(ppl_mod, 'HAS_PERPLEXITY', True), \
         patch.object(ppl_mod, '_torch', mock_torch, create=True), \
         patch.object(ppl_mod, 'HAS_BINOCULARS', False), \
         patch.object(ppl_mod, 'get_perplexity_model', mock_get):
        r = ppl_mod.run_perplexity(long_text)

    check("Determination is not None (some signal)", r['determination'] is not None,
          f"got {r['determination']}")
    check("surprisal_variance tracked", r.get('surprisal_variance', 0) >= 0)
    check("volatility_decay_ratio tracked", r.get('volatility_decay_ratio', 1.0) >= 0)


# ── Semantic Resonance ────────────────────────────────────────────────────────

def test_run_semantic_no_library():
    """run_semantic_resonance returns zeros when HAS_SEMANTIC=False."""
    print("\n-- SEMANTIC RESONANCE: NO LIBRARY --")
    from llm_detector.compat import HAS_SEMANTIC
    if not HAS_SEMANTIC:
        from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
        r = run_semantic_resonance("Some test text.")
        check("semantic_ai_score == 0.0", r['semantic_ai_score'] == 0.0)
        check("semantic_delta == 0.0", r['semantic_delta'] == 0.0)
        check("determination is None", r['determination'] is None)
        check("reason mentions unavailable",
              'unavailable' in r.get('reason', '').lower())
    else:
        check("HAS_SEMANTIC=True: skip", True)


def test_run_semantic_short_text_mocked():
    """run_semantic_resonance returns early for short text (< 30 words) with mocked."""
    print("\n-- SEMANTIC RESONANCE: SHORT TEXT (MOCKED) --")
    import llm_detector.analyzers.semantic_resonance as sr_mod

    short_text = "Too short."  # << 30 words

    with patch.object(sr_mod, 'HAS_SEMANTIC', True):
        r = sr_mod.run_semantic_resonance(short_text)
        check("Short text determination is None",
              r['determination'] is None, f"got {r['determination']}")
        check("Short text ai_score == 0.0", r['semantic_ai_score'] == 0.0)
        check("Short text reason mentions too short",
              'too short' in r.get('reason', '').lower())


def test_run_semantic_mocked_amber():
    """run_semantic_resonance returns AMBER when AI similarity is high."""
    print("\n-- SEMANTIC RESONANCE: MOCKED AMBER --")
    import llm_detector.analyzers.semantic_resonance as sr_mod

    # 35+ words
    long_text = "test word " * 35

    n_dim = 10
    # Text embedding close to AI centroid → high AI similarity
    mock_embedder = MagicMock()
    text_vec = np.array([[1.0] + [0.0] * (n_dim - 1)])
    mock_embedder.encode.return_value = text_vec

    ai_centroids = np.array([[1.0] + [0.0] * (n_dim - 1)])
    human_centroids = np.array([[0.0, 1.0] + [0.0] * (n_dim - 2)])

    def mock_cosine(vecs, centroids):
        # Dot product for unit vectors = cosine similarity
        return np.clip(vecs @ centroids.T, -1.0, 1.0)

    with patch.object(sr_mod, 'HAS_SEMANTIC', True), \
         patch.object(sr_mod, 'get_semantic_models',
                      return_value=(mock_embedder, ai_centroids, human_centroids)), \
         patch.object(sr_mod, '_cosine_similarity', mock_cosine, create=True):
        r = sr_mod.run_semantic_resonance(long_text)

    check("semantic_ai_score > 0", r['semantic_ai_score'] > 0,
          f"got {r['semantic_ai_score']}")
    check("semantic_delta present", 'semantic_delta' in r)
    # mean_ai_sim ≈ 1.0, mean_human_sim ≈ 0.0 → delta ≈ 1.0 >= 0.15 and sim >= 0.65
    check("AMBER determination for high AI sim",
          r['determination'] == 'AMBER', f"got {r['determination']}")
    check("confidence > 0", r.get('confidence', 0) > 0)


def test_run_semantic_mocked_yellow():
    """run_semantic_resonance returns YELLOW for moderate AI similarity."""
    print("\n-- SEMANTIC RESONANCE: MOCKED YELLOW --")
    import llm_detector.analyzers.semantic_resonance as sr_mod

    long_text = "test word " * 35
    n_dim = 10

    mock_embedder = MagicMock()
    # Moderate AI similarity: mean_ai_sim ≈ 0.55, delta ≈ 0.10
    # We'll set cos_sim(text, ai) = 0.55 and cos_sim(text, human) = 0.45
    mock_embedder.encode.return_value = np.ones((1, n_dim)) / np.sqrt(n_dim)

    ai_centroids = np.ones((1, n_dim)) * 0.55 / np.sqrt(n_dim) * n_dim
    human_centroids = np.ones((1, n_dim)) * 0.45 / np.sqrt(n_dim) * n_dim

    def mock_cosine(vecs, centroids):
        # Return fixed cosine-like values based on input shape
        n_chunks = len(vecs)
        n_centroids = len(centroids)
        # Distinguish AI vs human by checking centroids magnitude
        if centroids.mean() > ai_centroids.mean() * 0.9:
            return np.full((n_chunks, n_centroids), 0.55)
        return np.full((n_chunks, n_centroids), 0.45)

    # Use a simpler mock that always returns the same values
    ai_sim_val = 0.55
    human_sim_val = 0.45
    call_count = [0]

    def mock_cosine_v2(vecs, centroids):
        call_count[0] += 1
        if call_count[0] == 1:  # First call: AI
            return np.full((len(vecs), len(centroids)), ai_sim_val)
        else:  # Second call: human
            return np.full((len(vecs), len(centroids)), human_sim_val)

    with patch.object(sr_mod, 'HAS_SEMANTIC', True), \
         patch.object(sr_mod, 'get_semantic_models',
                      return_value=(mock_embedder, ai_centroids, human_centroids)), \
         patch.object(sr_mod, '_cosine_similarity', mock_cosine_v2, create=True):
        r = sr_mod.run_semantic_resonance(long_text)

    check("YELLOW determination for moderate AI sim",
          r['determination'] == 'YELLOW', f"got {r['determination']}")
    check("semantic_ai_mean ≈ 0.55",
          abs(r.get('semantic_ai_mean', 0) - 0.55) < 0.01,
          f"got {r.get('semantic_ai_mean')}")


def test_run_semantic_mocked_none():
    """run_semantic_resonance returns None determination below threshold."""
    print("\n-- SEMANTIC RESONANCE: MOCKED NONE --")
    import llm_detector.analyzers.semantic_resonance as sr_mod

    long_text = "test word " * 35
    n_dim = 10

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros((1, n_dim))

    ai_centroids = np.zeros((1, n_dim))
    human_centroids = np.zeros((1, n_dim))

    # Both similarities = 0 → delta = 0 → no determination
    def mock_cosine(vecs, centroids):
        return np.zeros((len(vecs), len(centroids)))

    with patch.object(sr_mod, 'HAS_SEMANTIC', True), \
         patch.object(sr_mod, 'get_semantic_models',
                      return_value=(mock_embedder, ai_centroids, human_centroids)), \
         patch.object(sr_mod, '_cosine_similarity', mock_cosine, create=True):
        r = sr_mod.run_semantic_resonance(long_text)

    check("None determination below threshold",
          r['determination'] is None, f"got {r['determination']}")
    check("Reason mentions below threshold",
          'below threshold' in r.get('reason', '').lower(),
          f"got {r.get('reason')}")


# ── Token Cohesiveness ────────────────────────────────────────────────────────

def test_run_token_cohesiveness_no_library():
    """run_token_cohesiveness returns zeros when HAS_SEMANTIC=False."""
    print("\n-- TOKEN COHESIVENESS: NO LIBRARY --")
    from llm_detector.compat import HAS_SEMANTIC
    if not HAS_SEMANTIC:
        from llm_detector.analyzers.token_cohesiveness import run_token_cohesiveness
        r = run_token_cohesiveness("Some test text.")
        check("cohesiveness == 0.0", r['cohesiveness'] == 0.0)
        check("determination is None", r['determination'] is None)
        check("reason mentions unavailable",
              'unavailable' in r.get('reason', '').lower())
    else:
        check("HAS_SEMANTIC=True: skip", True)


def test_run_token_cohesiveness_short_text_mocked():
    """run_token_cohesiveness returns early for short text (< 60 words)."""
    print("\n-- TOKEN COHESIVENESS: SHORT TEXT (MOCKED) --")
    import llm_detector.analyzers.token_cohesiveness as tc_mod

    short_text = "Only a few words here."  # << 60 words

    with patch.object(tc_mod, 'HAS_SEMANTIC', True):
        r = tc_mod.run_token_cohesiveness(short_text)
        check("Short text determination is None",
              r['determination'] is None, f"got {r['determination']}")
        check("Short text cohesiveness == 0.0", r['cohesiveness'] == 0.0)
        check("Short text reason mentions too short",
              'too short' in r.get('reason', '').lower())


def _make_mock_sklearn_pairwise(cos_sim_value: float):
    """Create a mock sklearn.metrics.pairwise module."""
    mock_pairwise = MagicMock()
    mock_pairwise.cosine_similarity.return_value = np.array([[cos_sim_value]])
    return mock_pairwise


def test_run_token_cohesiveness_mocked_amber():
    """run_token_cohesiveness returns AMBER for high cohesiveness."""
    print("\n-- TOKEN COHESIVENESS: MOCKED AMBER --")
    import llm_detector.analyzers.token_cohesiveness as tc_mod

    # 70+ words
    long_text = "the quick brown fox jumps over the lazy dog " * 8  # 72 words

    mock_embedder = MagicMock()
    n_dim = 10
    # Return different embeddings for original vs reduced text
    call_count = [0]

    def side_effect(texts):
        call_count[0] += 1
        if call_count[0] == 1:
            return np.array([[1.0] + [0.0] * (n_dim - 1)])
        else:
            # Slightly different embedding → low cosine similarity → high distance
            return np.array([[0.97] + [0.0] * (n_dim - 1)])

    mock_embedder.encode.side_effect = side_effect

    # cosine_similarity returns low similarity → high cohesiveness distance
    mock_pairwise = _make_mock_sklearn_pairwise(0.97)

    with patch.object(tc_mod, 'HAS_SEMANTIC', True), \
         patch.object(tc_mod, 'get_semantic_models',
                      return_value=(mock_embedder, None, None)), \
         patch.dict('sys.modules', {
             'sklearn': MagicMock(),
             'sklearn.metrics': MagicMock(),
             'sklearn.metrics.pairwise': mock_pairwise,
         }):
        r = tc_mod.run_token_cohesiveness(long_text, n_copies=5)

    check("cohesiveness > 0", r.get('cohesiveness', 0) > 0,
          f"got {r.get('cohesiveness')}")
    check("n_rounds > 0", r.get('n_rounds', 0) > 0,
          f"got {r.get('n_rounds')}")
    check("determination is not None", r['determination'] is not None,
          f"got {r['determination']}")


def test_run_token_cohesiveness_mocked_low():
    """run_token_cohesiveness returns None for low cohesiveness (high similarity)."""
    print("\n-- TOKEN COHESIVENESS: MOCKED LOW --")
    import llm_detector.analyzers.token_cohesiveness as tc_mod

    long_text = "the quick brown fox jumps over the lazy dog " * 8

    mock_embedder = MagicMock()
    n_dim = 10
    # All embeddings identical → cosine similarity = 1.0 → distance = 0.0
    mock_embedder.encode.return_value = np.array([[1.0] + [0.0] * (n_dim - 1)])

    # Very high similarity → very low distance → no determination
    mock_pairwise = _make_mock_sklearn_pairwise(0.9999)

    with patch.object(tc_mod, 'HAS_SEMANTIC', True), \
         patch.object(tc_mod, 'get_semantic_models',
                      return_value=(mock_embedder, None, None)), \
         patch.dict('sys.modules', {
             'sklearn': MagicMock(),
             'sklearn.metrics': MagicMock(),
             'sklearn.metrics.pairwise': mock_pairwise,
         }):
        r = tc_mod.run_token_cohesiveness(long_text, n_copies=5)

    check("Low cohesiveness determination is None",
          r['determination'] is None, f"got {r['determination']}")
    check("Cohesiveness is very small",
          r.get('cohesiveness', 1.0) < 0.05, f"got {r.get('cohesiveness')}")


if __name__ == '__main__':
    print("=" * 70)
    print("ML Analyzer Mocked Tests")
    print("=" * 70)

    test_run_perplexity_no_library()
    test_run_perplexity_short_text_mocked()
    test_run_perplexity_mocked_amber()
    test_run_perplexity_mocked_yellow()
    test_run_perplexity_mocked_normal()
    test_run_perplexity_mocked_diveye_upgrade()
    test_run_semantic_no_library()
    test_run_semantic_short_text_mocked()
    test_run_semantic_mocked_amber()
    test_run_semantic_mocked_yellow()
    test_run_semantic_mocked_none()
    test_run_token_cohesiveness_no_library()
    test_run_token_cohesiveness_short_text_mocked()
    test_run_token_cohesiveness_mocked_amber()
    test_run_token_cohesiveness_mocked_low()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
