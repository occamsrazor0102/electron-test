"""Shared text utilities used across multiple modules."""

import re
from llm_detector.compat import HAS_SPACY, get_nlp

# Top-50 English function words (closed class, highly stable across registers)
ENGLISH_FUNCTION_WORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'may', 'might', 'can', 'could', 'must',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'about', 'between', 'through', 'after', 'before',
    'and', 'or', 'but', 'not', 'if', 'that', 'this', 'it', 'he', 'she',
    'they', 'we', 'i', 'you', 'my', 'your', 'his', 'her', 'its', 'our',
    'their', 'who', 'which', 'what', 'there',
])


def type_token_ratio(tokens):
    """Type-Token Ratio: vocabulary richness (unique tokens / total tokens)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def get_sentences(text):
    """Segment text into sentences using spacy sentencizer or regex fallback."""
    if HAS_SPACY:
        nlp = get_nlp()
        doc = nlp(text)
        return [s.text for s in doc.sents]
    else:
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sents if s.strip()]


def get_sentence_spans(text):
    """Segment text into sentences with character offsets.

    Returns list of (sentence_text, start_char, end_char) tuples.
    """
    if HAS_SPACY:
        nlp = get_nlp()
        doc = nlp(text)
        return [(s.text, s.start_char, s.end_char) for s in doc.sents]
    else:
        spans = []
        for m in re.finditer(r'[^.!?]*[.!?]+\s*|[^.!?]+$', text):
            t = m.group().strip()
            if t:
                spans.append((t, m.start(), m.start() + len(t)))
        if not spans and text.strip():
            spans.append((text.strip(), 0, len(text.strip())))
        return spans
