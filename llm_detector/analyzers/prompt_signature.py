"""Prompt-engineering signature detection -- CFD, MFSR, numbered criteria."""

import re
from llm_detector.text_utils import get_sentences

CONSTRAINT_FRAMES = [
    r'must account for', r'should be visible', r'at least \d+[%$]?',
    r'at or below', r'no more than', r'no \w+ may', r'must have',
    r'should address', r'should be delivered', r'within \d+%',
    r'or higher', r'or lower', r'instead of', r'without exceeding',
    r'about \d+[-–]\d+', r'strictly on',
    r'must include\b', r'must address\b', r'must be \w+',
    r'you may not\b',
    r'may not (?:be|introduce|omit|use|exceed|include)\b',
    r'in this exact\b', r'with exactly \d+',
    r'every \w+ must\b', r'all \w+ must\b',
    r'clearly (?:list|state|describe|identify|document)',
    r'(?:document|report|response) must\b',
    r'(?:following|these) sections',
    r'use \w+ formatting', r'plain language',
    r'no \w+[- ]only\b',
]

META_DESIGN_PATTERNS = [
    r'(?i)workflows? tested',
    r'(?i)acceptance (checklist|criteria)',
    r'(?i)(used for|for) grading',
    r'(?i)SOC \d{2}-?\d{4}',
    r'(?i)expected effort:?\s*\d',
    r'(?i)deliberate (anomalies|errors|issues)',
    r'(?i)checkable artifacts',
    r'(?i)authoritative source of truth',
    r'(?i)scenario anchor date',
    r'(?i)avoid vague language',
    r'(?i)explicit non-functional',
    r'(?i)grounded in\b',
]

# Pre-compiled versions to avoid re-compiling on every call.
_CONSTRAINT_FRAMES_RE = [re.compile(pat, re.IGNORECASE) for pat in CONSTRAINT_FRAMES]
_META_DESIGN_RE = [re.compile(pat) for pat in META_DESIGN_PATTERNS]


def run_prompt_signature(text):
    """Detect prompt-engineering signatures. Returns dict of metrics."""
    sents = get_sentences(text)
    n_sents = max(len(sents), 1)
    word_count = len(text.split())

    total_frames = 0
    distinct_pats = set()
    for pat, compiled_pat in zip(CONSTRAINT_FRAMES, _CONSTRAINT_FRAMES_RE):
        matches = compiled_pat.findall(text)
        if matches:
            total_frames += len(matches)
            distinct_pats.add(pat)
    cfd = total_frames / n_sents

    multi_frame = 0
    for sent in sents:
        ct = sum(1 for compiled_pat in _CONSTRAINT_FRAMES_RE if compiled_pat.search(sent))
        if ct >= 2:
            multi_frame += 1
    mfsr = multi_frame / n_sents

    has_role = bool(re.search(r'you (are|work|supervise|manage|serve|lead|oversee)', text[:600], re.IGNORECASE))
    has_deliverable = bool(re.search(r'(submit|deliver|present|provide|create|produce|prepare|generate)\s+(your|the|a|an|exactly)', text, re.IGNORECASE))
    has_closing = bool(re.search(r'(final|should be delivered|all conclusions|base all|submission|deliverable)', text[-300:], re.IGNORECASE))
    fc = int(has_role) + int(has_deliverable) + int(has_closing)

    cond_count = len(re.findall(r'\bif\b[^.]*?,', text, re.IGNORECASE))
    cond_count += len(re.findall(r'\bwhen\b[^.]*?,', text, re.IGNORECASE))
    cond_count += len(re.findall(r'\bunless\b', text, re.IGNORECASE))
    cond_density = cond_count / n_sents

    meta_hits = [pat for pat, compiled_pat in zip(META_DESIGN_PATTERNS, _META_DESIGN_RE)
                 if compiled_pat.search(text)]

    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|s|d|ll|m)\b", text, re.IGNORECASE))

    must_count = len(re.findall(r'\bmust\b', text, re.IGNORECASE))
    must_rate = must_count / n_sents

    numbered_criteria = len(re.findall(r'^\s*\d{1,2}[.)]\s+.{20,}', text, re.MULTILINE))

    composite = 0.0
    if cfd >= 0.50: composite += 0.40
    elif cfd >= 0.30: composite += 0.25
    elif cfd >= 0.15: composite += 0.10
    if len(distinct_pats) >= 8: composite += 0.20
    elif len(distinct_pats) >= 5: composite += 0.12
    elif len(distinct_pats) >= 3: composite += 0.05
    if len(meta_hits) >= 3: composite += 0.20
    elif len(meta_hits) >= 1: composite += 0.08
    if fc == 3: composite += 0.10
    if fc >= 2 and len(distinct_pats) >= 8:
        composite += 0.15
    if numbered_criteria >= 15: composite += 0.15
    elif numbered_criteria >= 10: composite += 0.08
    if contractions == 0 and word_count > 500: composite += 0.05

    return {
        'cfd': cfd,
        'distinct_frames': len(distinct_pats),
        'mfsr': mfsr,
        'framing_completeness': fc,
        'conditional_density': cond_density,
        'meta_design_hits': len(meta_hits),
        'meta_design_details': meta_hits,
        'contractions': contractions,
        'must_count': must_count,
        'must_rate': must_rate,
        'numbered_criteria': numbered_criteria,
        'composite': min(composite, 1.0),
    }
