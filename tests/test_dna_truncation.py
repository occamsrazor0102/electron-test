"""Tests for structure-aware _dna_truncate_text and _detect_text_format."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.analyzers.continuation_api import (
    _dna_truncate_text, _detect_text_format, _format_hint_str,
)


# ── _detect_text_format ────────────────────────────────────────────────


def test_detect_format_empty():
    assert _detect_text_format("no special structure here") == []


def test_detect_format_paragraph_breaks():
    text = "First paragraph here.\n\nSecond paragraph here."
    features = _detect_text_format(text)
    assert "paragraph breaks" in features


def test_detect_format_numbered_sections():
    text = "Introduction.\n1. First point\n2. Second point"
    features = _detect_text_format(text)
    assert "numbered sections" in features


def test_detect_format_bullet_points():
    text = "Items:\n- Apple\n- Banana\n- Cherry"
    features = _detect_text_format(text)
    assert "bullet points" in features


def test_detect_format_markdown_headings():
    text = "# Title\n## Section\nContent here."
    features = _detect_text_format(text)
    assert "markdown headings" in features


def test_detect_format_multiple():
    text = "# Title\n\n1. First point\n\n- bullet"
    features = _detect_text_format(text)
    assert "paragraph breaks" in features
    assert "numbered sections" in features
    assert "bullet points" in features
    assert "markdown headings" in features


# ── _format_hint_str ───────────────────────────────────────────────────


def test_format_hint_empty():
    assert _format_hint_str([]) == ""


def test_format_hint_single():
    result = _format_hint_str(["paragraph breaks"])
    assert result == " It uses paragraph breaks."


def test_format_hint_two_features():
    result = _format_hint_str(["paragraph breaks", "bullet points"])
    assert "paragraph breaks" in result
    assert "bullet points" in result
    assert " and " in result


def test_format_hint_three_features():
    result = _format_hint_str(["paragraph breaks", "numbered sections", "bullet points"])
    assert "paragraph breaks" in result
    assert "numbered sections" in result
    assert "bullet points" in result
    assert " and bullet points." in result


# ── _dna_truncate_text ─────────────────────────────────────────────────


def test_truncate_returns_tuple():
    text = "Hello world. " * 50
    prefix, continuation = _dna_truncate_text(text)
    assert isinstance(prefix, str) and isinstance(continuation, str)
    assert len(prefix) > 0 and len(continuation) > 0


def test_truncate_paragraph_boundary():
    """With >= 3 paragraphs, truncation should snap to a paragraph boundary."""
    para_a = "This is the first paragraph. " * 15
    para_b = "This is the second paragraph. " * 15
    para_c = "This is the third paragraph. " * 15
    para_d = "This is the fourth paragraph. " * 15
    paragraphs = [para_a, para_b, para_c, para_d]
    text = "\n\n".join(paragraphs)

    prefix, continuation = _dna_truncate_text(text, ratio=0.5)

    # The result must cover the full text content (modulo whitespace normalisation)
    prefix_words = set(prefix.lower().split())
    continuation_words = set(continuation.lower().split())
    # Every token in the original must appear in one of the two halves
    original_words = set(text.lower().split())
    assert original_words == prefix_words | continuation_words

    # The continuation must meet the minimum-words requirement
    assert len(continuation.split()) >= 30


def test_truncate_continuation_min_words():
    """Continuation must always have >= 30 words when the full text is long enough."""
    words = ["word"] * 200
    text = " ".join(words)
    prefix, continuation = _dna_truncate_text(text, ratio=0.5)
    assert len(continuation.split()) >= 30


def test_truncate_short_text_word_fallback():
    """Very short text (< 4 sentences, no paragraphs) uses word-level fallback."""
    text = "Short text here okay."
    prefix, continuation = _dna_truncate_text(text, ratio=0.5)
    assert isinstance(prefix, str)
    assert isinstance(continuation, str)
    # Together they should cover all words
    combined_words = prefix.split() + continuation.split()
    assert len(combined_words) == len(text.split())


def test_truncate_ratio_affects_split_point():
    """Higher ratio should yield a longer prefix."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. " * 10
    prefix_low, _ = _dna_truncate_text(text, ratio=0.3)
    prefix_high, _ = _dna_truncate_text(text, ratio=0.7)
    assert len(prefix_low.split()) < len(prefix_high.split())


def test_truncate_sentence_fallback():
    """Text without paragraph breaks but >= 4 sentences uses sentence-level split."""
    sentences = [f"This is sentence number {i} in the test." for i in range(20)]
    text = " ".join(sentences)
    prefix, continuation = _dna_truncate_text(text, ratio=0.5)
    assert len(prefix.split()) > 0
    assert len(continuation.split()) >= 30


def test_truncate_paragraph_continuation_sufficient():
    """Falls back to sentence split if paragraph continuation is too short."""
    para_a = "Long first paragraph with many words. " * 20
    para_b = "Short para."  # too short to satisfy >= 30 words in continuation
    para_c = "Long last paragraph with many words. " * 20
    text = f"{para_a}\n\n{para_b}\n\n{para_c}"
    # With ratio=0.98, paragraph path would yield a very short continuation —
    # the function should fall back gracefully.
    prefix, continuation = _dna_truncate_text(text, ratio=0.98)
    assert isinstance(prefix, str) and isinstance(continuation, str)
