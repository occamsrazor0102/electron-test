"""Tests for MATTR and perplexity burstiness features."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_mattr_basic():
    print("\n-- MATTR: basic computation --")
    from llm_detector.analyzers.stylometry import _mattr

    # Repeated words → low TTR but MATTR should also be low
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat'] * 20  # 120 words, low diversity
    val = _mattr(words, window=50)
    check("MATTR returns float", isinstance(val, float))
    check("MATTR in [0, 1]", 0 <= val <= 1, f"got {val}")
    check("Low diversity → low MATTR", val < 0.5, f"got {val}")


def test_mattr_high_diversity():
    print("\n-- MATTR: high diversity text --")
    from llm_detector.analyzers.stylometry import _mattr

    # All unique words → high MATTR
    words = [f'word{i}' for i in range(200)]
    val = _mattr(words, window=50)
    check("High diversity → high MATTR", val > 0.9, f"got {val}")


def test_mattr_short_fallback():
    print("\n-- MATTR: short text fallback --")
    from llm_detector.analyzers.stylometry import _mattr

    # Fewer words than window → fallback to simple TTR
    words = ['hello', 'world', 'foo', 'bar', 'hello']
    val = _mattr(words, window=50)
    expected = len(set(words)) / len(words)
    check("Short text uses simple TTR", abs(val - expected) < 0.01,
          f"got {val}, expected ~{expected}")


def test_mattr_in_stylometric_features():
    print("\n-- MATTR: included in extract_stylometric_features --")
    from llm_detector.analyzers.stylometry import extract_stylometric_features

    text = "The quick brown fox jumps over the lazy dog. " * 15
    features = extract_stylometric_features(text)
    check("'mattr' key present", 'mattr' in features, f"keys: {list(features.keys())}")
    check("MATTR is float", isinstance(features['mattr'], float))
    check("MATTR in [0, 1]", 0 <= features['mattr'] <= 1, f"got {features['mattr']}")
    # Standard TTR should also still be present
    check("'type_token_ratio' still present", 'type_token_ratio' in features)


def test_mattr_window_independence():
    """MATTR should be more stable across different text lengths than TTR."""
    print("\n-- MATTR: length independence vs TTR --")
    from llm_detector.analyzers.stylometry import _mattr, extract_stylometric_features

    base = "The quick brown fox jumps over the lazy dog near the river bank. "
    short_text = base * 5   # ~65 words
    long_text = base * 30   # ~390 words

    feat_short = extract_stylometric_features(short_text)
    feat_long = extract_stylometric_features(long_text)

    ttr_diff = abs(feat_short['type_token_ratio'] - feat_long['type_token_ratio'])
    mattr_diff = abs(feat_short['mattr'] - feat_long['mattr'])

    check("MATTR more stable than TTR across lengths",
          mattr_diff <= ttr_diff + 0.01,  # small tolerance
          f"MATTR diff={mattr_diff:.4f}, TTR diff={ttr_diff:.4f}")


def test_ppl_burstiness_in_empty_result():
    print("\n-- PERPLEXITY BURSTINESS: present in empty result --")
    from llm_detector.analyzers.perplexity import _PPL_EMPTY

    check("ppl_burstiness in _PPL_EMPTY", 'ppl_burstiness' in _PPL_EMPTY)
    check("sentence_ppl_count in _PPL_EMPTY", 'sentence_ppl_count' in _PPL_EMPTY)
    check("ppl_burstiness defaults to 0", _PPL_EMPTY['ppl_burstiness'] == 0.0)


def test_ppl_burstiness_no_perplexity():
    print("\n-- PERPLEXITY BURSTINESS: graceful without transformers --")
    from unittest.mock import patch

    with patch('llm_detector.analyzers.perplexity.HAS_PERPLEXITY', False):
        from llm_detector.analyzers.perplexity import run_perplexity
        result = run_perplexity("Some text here that is long enough to test.")
        check("ppl_burstiness present", 'ppl_burstiness' in result)
        check("ppl_burstiness is 0", result['ppl_burstiness'] == 0.0)
        check("sentence_ppl_count is 0", result['sentence_ppl_count'] == 0)


if __name__ == '__main__':
    print("=" * 70)
    print("MATTR & Perplexity Burstiness Tests")
    print("=" * 70)

    test_mattr_basic()
    test_mattr_high_diversity()
    test_mattr_short_fallback()
    test_mattr_in_stylometric_features()
    test_mattr_window_independence()
    test_ppl_burstiness_in_empty_result()
    test_ppl_burstiness_no_perplexity()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
