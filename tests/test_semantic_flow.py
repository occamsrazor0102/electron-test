"""Tests for the semantic flow analyzer (inter-sentence variance)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import numpy as np

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


def test_short_text_graceful_skip():
    print("\n-- SEMANTIC FLOW: short text graceful skip --")
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    # Mock HAS_SEMANTIC to True but provide short text
    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', True):
        with patch('llm_detector.analyzers.semantic_flow.get_sentences', return_value=['Short.']):
            result = run_semantic_flow("Short.")
            check("Short text returns None determination", result['determination'] is None)
            check("Short text has 0 variance", result['flow_variance'] == 0.0)
            check("Reason mentions too few", 'too few' in result['reason'])


def test_no_semantic_libs():
    print("\n-- SEMANTIC FLOW: no sentence-transformers --")
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', False):
        result = run_semantic_flow("Some text here.")
        check("No libs returns None determination", result['determination'] is None)
        check("Reason mentions unavailable", 'unavailable' in result['reason'])
        check("flow_variance is 0", result['flow_variance'] == 0.0)


def test_uniform_text_low_variance():
    """Simulates AI text: uniform embeddings → low variance."""
    print("\n-- SEMANTIC FLOW: uniform text (AI-like) --")
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    # Create mock embedder that returns very similar embeddings
    mock_embedder = MagicMock()
    base = np.random.RandomState(42).randn(384)
    base = base / np.linalg.norm(base)

    def mock_encode(sentences):
        # All sentences get nearly identical embeddings (tiny noise)
        rng = np.random.RandomState(0)
        return np.array([base + rng.randn(384) * 0.01 for _ in sentences])

    mock_embedder.encode = mock_encode

    sentences = [
        'This is the first sentence about the topic.',
        'This is the second sentence continuing the topic.',
        'This is the third sentence building on the topic.',
        'This is the fourth sentence expanding the topic.',
        'This is the fifth sentence concluding the topic.',
        'This is the sixth sentence summarizing the topic.',
    ]

    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', True):
        with patch('llm_detector.analyzers.semantic_flow.get_sentences', return_value=sentences):
            with patch('llm_detector.analyzers.semantic_flow.get_semantic_models', return_value=(mock_embedder, None, None)):
                result = run_semantic_flow(' '.join(sentences))
                check("Low variance detected", result['flow_variance'] < 0.01,
                      f"got variance={result['flow_variance']}")
                check("High mean similarity", result['flow_mean'] > 0.9,
                      f"got mean={result['flow_mean']}")
                check("n_sentences correct", result['n_sentences'] == 6)
                check("Has similarities list", len(result['flow_similarities']) == 5)


def test_varied_text_high_variance():
    """Simulates human text: diverse embeddings → high variance."""
    print("\n-- SEMANTIC FLOW: varied text (human-like) --")
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    mock_embedder = MagicMock()

    def mock_encode(sentences):
        # Create embeddings with highly variable consecutive similarities:
        # pairs (0,1) very similar, (1,2) very different, etc.
        rng = np.random.RandomState(42)
        base = rng.randn(384)
        base = base / np.linalg.norm(base)
        embeddings = []
        for i in range(len(sentences)):
            if i % 2 == 0:
                v = base + rng.randn(384) * 0.05  # close to base
            else:
                v = rng.randn(384)  # random direction
            v = v / np.linalg.norm(v)
            embeddings.append(v)
        return np.array(embeddings)

    mock_embedder.encode = mock_encode

    sentences = [
        'The weather today is sunny and warm.',
        'Programming in Python requires patience.',
        'My cat jumped onto the refrigerator.',
        'Mathematics reveals hidden patterns.',
        'The concert last night was incredible.',
        'Quantum physics challenges intuition.',
    ]

    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', True):
        with patch('llm_detector.analyzers.semantic_flow.get_sentences', return_value=sentences):
            with patch('llm_detector.analyzers.semantic_flow.get_semantic_models', return_value=(mock_embedder, None, None)):
                result = run_semantic_flow(' '.join(sentences))
                check("Higher variance than uniform text", result['flow_variance'] > 0.001,
                      f"got variance={result['flow_variance']}")
                check("No AI determination", result['determination'] is None,
                      f"got {result['determination']}")


def test_result_dict_keys():
    print("\n-- SEMANTIC FLOW: result dict completeness --")
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', False):
        result = run_semantic_flow("text")
        expected_keys = {'flow_similarities', 'flow_mean', 'flow_variance', 'flow_std',
                         'n_sentences', 'determination', 'confidence', 'reason'}
        check("All expected keys present", expected_keys.issubset(set(result.keys())),
              f"missing: {expected_keys - set(result.keys())}")


def test_zero_vectors_safe_similarity():
    print("\n-- SEMANTIC FLOW: zero-vector embeddings fallback --")
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    mock_embedder = MagicMock()
    mock_embedder.encode = MagicMock(return_value=np.zeros((3, 384)))

    sentences = ['s1', 's2', 's3']

    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', True):
        with patch('llm_detector.analyzers.semantic_flow.get_sentences', return_value=sentences):
            with patch('llm_detector.analyzers.semantic_flow.get_semantic_models', return_value=(mock_embedder, None, None)):
                result = run_semantic_flow(' '.join(sentences), min_sentences=3)
                check("Zero vectors handled without NaN", all(s == 0.0 for s in result['flow_similarities']),
                      f"got {result['flow_similarities']}")
                check("Variance zero for zero vectors", result['flow_variance'] == 0.0,
                      f"got variance={result['flow_variance']}")
                check("Determination remains None", result['determination'] is None)


def test_similarities_are_native_float():
    """Verify that similarities list contains native Python floats (not numpy scalars).

    The statistics module on Python 3.9/3.10 calls _exact_ratio which accesses
    .numerator — an attribute absent on numpy.float64.  Wrapping _cosine output
    in float() prevents this.
    """
    print("\n-- SEMANTIC FLOW: similarities are native Python float --")
    import statistics as _stat
    from llm_detector.analyzers.semantic_flow import run_semantic_flow

    mock_embedder = MagicMock()
    base = np.random.RandomState(42).randn(384)
    base = base / np.linalg.norm(base)

    def mock_encode(sentences):
        rng = np.random.RandomState(0)
        return np.array([base + rng.randn(384) * 0.05 for _ in sentences])

    mock_embedder.encode = mock_encode

    sentences = ['A.', 'B.', 'C.', 'D.', 'E.']

    with patch('llm_detector.analyzers.semantic_flow.HAS_SEMANTIC', True):
        with patch('llm_detector.analyzers.semantic_flow.get_sentences', return_value=sentences):
            with patch('llm_detector.analyzers.semantic_flow.get_semantic_models', return_value=(mock_embedder, None, None)):
                result = run_semantic_flow(' '.join(sentences))
                for s in result['flow_similarities']:
                    check("similarity is native float", type(s) is float,
                          f"got {type(s).__name__}")
                # Verify statistics.mean works without AttributeError
                _stat.mean(result['flow_similarities'])
                check("statistics.mean works on similarities", True)


if __name__ == '__main__':
    print("=" * 70)
    print("Semantic Flow Analyzer Tests")
    print("=" * 70)

    test_short_text_graceful_skip()
    test_no_semantic_libs()
    test_uniform_text_low_variance()
    test_varied_text_high_variance()
    test_result_dict_keys()
    test_zero_vectors_safe_similarity()
    test_similarities_are_native_float()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
