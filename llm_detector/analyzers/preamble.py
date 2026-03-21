"""Preamble detection -- catches LLM output artifacts.

Detects: assistant acknowledgments, artifact delivery frames, first-person
creation claims, meta-design language, style masking, editorial meta-commentary,
and Chain-of-Thought leakage from Large Reasoning Models (DeepSeek-R1, o1/o3).
"""

import re

PREAMBLE_PATTERNS = [
    (r"(?i)^\s*[\"']?(got it|sure thing|absolutely|certainly|of course)[.!,\s]", "assistant_ack", "CRITICAL"),
    (r"(?i)^\s*[\"']?here(?:'s| is| are)\s+(your|the|a)\s+(final|updated|revised|complete|rewritten|prompt|task|evaluation)", "artifact_delivery", "CRITICAL"),
    (r"(?i)^\s*[\"']?below is\s+(a\s+)?(rewritten|revised|updated|the|your)", "artifact_delivery", "CRITICAL"),
    (r"(?i)(copy[- ]?paste|ready to use|plug[- ]and[- ]play)", "copy_paste_instruction", "MEDIUM"),
    (r"(?i)(failure[- ]inducing|designed to (test|challenge|trip|catch|induce))", "meta_design", "CRITICAL"),
    (r"(?i)^\s*[\"']?(I'?ve |I have |I'?ll |let me )(created?|drafted?|prepared?|written|designed|built|put together)", "first_person_creation", "CRITICAL"),
    (r"(?i)(natural workplace style|sounds? like a real|human[- ]issued|reads? like a human)", "style_masking", "HIGH"),
    (r"(?i)notes on what I (fixed|changed|cleaned|updated|revised)", "editorial_meta", "HIGH"),
    # Chain-of-thought leakage from Large Reasoning Models (DeepSeek-R1, o1/o3)
    (r"<think>", "cot_leakage", "CRITICAL"),
    (r"</think>", "cot_leakage", "CRITICAL"),
    (r"<reasoning>", "cot_leakage", "CRITICAL"),
    (r"</reasoning>", "cot_leakage", "CRITICAL"),
    (r"(?i)\blet me (?:rethink|reconsider|recalculate|re-examine|verify|double[- ]check)\b", "cot_reasoning", "HIGH"),
    (r"(?i)\bwait,?\s+(?:actually|no|let me|that'?s not)", "cot_self_correction", "HIGH"),
    (r"(?i)\bhmm,?\s+(?:let me|on second thought|actually)", "cot_self_correction", "HIGH"),
    (r"(?i)\bmy (?:final|revised|updated) answer (?:is|should|would)\b", "cot_conclusion", "HIGH"),
    (r"(?i)\bstep \d+\s*:", "cot_step_numbering", "MEDIUM"),
]

# Pre-compiled patterns with a flag indicating whether to search only the first
# 500 characters.  Compiling once at import time avoids repeated compilation on
# every run_preamble() call.
_TRUNCATED_NAMES = frozenset(('assistant_ack', 'artifact_delivery', 'first_person_creation', 'cot_leakage'))
_PREAMBLE_COMPILED = [
    (re.compile(pat), name, sev, name in _TRUNCATED_NAMES)
    for pat, name, sev in PREAMBLE_PATTERNS
]


def run_preamble(text):
    """Detect LLM preamble artifacts. Returns (score, severity, hits, spans)."""
    first_500 = text[:500]
    hits = []
    spans = []
    severity = 'NONE'

    for compiled_pat, name, sev, use_truncated in _PREAMBLE_COMPILED:
        search_text = first_500 if use_truncated else text
        match = compiled_pat.search(search_text)
        if match:
            hits.append((name, sev))
            spans.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group()[:80],
                'pattern': name,
                'severity': sev,
            })
            if sev == 'CRITICAL':
                severity = 'CRITICAL'
            elif sev == 'HIGH' and severity not in ('CRITICAL',):
                severity = 'HIGH'
            elif sev == 'MEDIUM' and severity == 'NONE':
                severity = 'MEDIUM'

    score = {'CRITICAL': 0.99, 'HIGH': 0.75, 'MEDIUM': 0.50, 'NONE': 0.0}[severity]
    return score, severity, hits, spans


def run_preamble_spans(text):
    """Return character-level spans for preamble pattern hits.

    Thin wrapper around run_preamble() for backward compatibility.
    Returns list of (start_char, end_char, matched_text, pattern_name, severity).
    """
    _, _, _, spans = run_preamble(text)
    return [(s['start'], s['end'], s['text'], s['pattern'], s['severity'])
            for s in spans]
