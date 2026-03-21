"""Intrinsic fingerprint detection -- LLM-preferred vocabulary."""

import re

FINGERPRINT_WORDS = [
    # Original 27 (ChatGPT-3.5 era, established in v0.51)
    'delve', 'utilize', 'comprehensive', 'streamline', 'leverage', 'robust',
    'facilitate', 'innovative', 'synergy', 'paradigm', 'holistic', 'nuanced',
    'multifaceted', 'spearhead', 'underscore', 'pivotal', 'landscape',
    'cutting-edge', 'actionable', 'seamlessly', 'noteworthy', 'meticulous',
    'endeavor', 'paramount', 'aforementioned', 'furthermore', 'henceforth',
    # v0.63 additions (Kobak et al. 2024 excess vocabulary, Science Advances)
    'tapestry', 'realm', 'embark', 'foster', 'showcasing',
]

_FINGERPRINT_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in FINGERPRINT_WORDS) + r')\b',
    re.IGNORECASE
)


def run_fingerprint(text):
    """Detect LLM fingerprint words. Returns (score, hit_count, rate)."""
    word_count = len(text.split())
    matches = _FINGERPRINT_RE.findall(text)
    hits = len(matches)
    rate = hits / max(word_count / 1000, 1)
    score = min(rate / 5.0, 1.0)
    return score, hits, rate


def run_fingerprint_spans(text):
    """Return character-level spans for fingerprint word hits.

    Returns list of (start_char, end_char, matched_text, 'fingerprint', word).
    """
    spans = []
    for m in _FINGERPRINT_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(), 'fingerprint', m.group().lower()))
    return spans
