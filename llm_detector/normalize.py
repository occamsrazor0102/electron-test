"""
Text normalization pre-pass.

Neutralizes cheap evasion attacks before all detection layers.
Ref: RAID benchmark (Dugan et al. 2024) — formatting perturbations,
     homoglyphs, and spacing attacks degrade metric-based detectors.
Ref: MGTBench (He et al. 2023) — paraphrasing sensitivity in rule-based
     detectors.
"""

import re
import unicodedata
from llm_detector.compat import HAS_FTFY

if HAS_FTFY:
    import ftfy

# Common homoglyph mappings: visually similar Unicode -> ASCII
_HOMOGLYPH_MAP = str.maketrans({
    '\u0410': 'A', '\u0412': 'B', '\u0421': 'C', '\u0415': 'E',
    '\u041d': 'H', '\u041a': 'K', '\u041c': 'M', '\u041e': 'O',
    '\u0420': 'P', '\u0422': 'T', '\u0425': 'X',
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x',
    '\u2018': "'", '\u2019': "'", '\u201a': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"',
    '\u2032': "'", '\u2033': '"',
    '\u2014': '--', '\u2013': '-', '\u2012': '-',
    '\u2026': '...', '\u22ef': '...',
    '\uff01': '!', '\uff1f': '?', '\uff0c': ',', '\uff0e': '.',
    '\uff1a': ':', '\uff1b': ';',
})

# Zero-width and invisible characters to strip
_INVISIBLE_RE = re.compile(
    '[\u200b\u200c\u200d\u200e\u200f'
    '\u2060\u2061\u2062\u2063\u2064'
    '\ufeff'
    '\u00ad'
    '\u034f'
    '\u180e'
    '\u2028\u2029'
    ']'
)

# Inter-character spacing: "l i k e  t h i s"
_INTERSPACED_RE = re.compile(r'(?<!\w)(\w) (\w) (\w) (\w)(?!\w)')


def normalize_text(text):
    """Normalize text to neutralize common evasion attacks.

    Returns (normalized_text, delta_report).
    """
    original = text
    original_len = max(len(text), 1)
    changes = 0
    ftfy_applied = False

    # 0. ftfy encoding repair
    if HAS_FTFY:
        pre_ftfy = text
        text = ftfy.fix_text(text)
        ftfy_changes = sum(1 for a, b in zip(pre_ftfy, text) if a != b)
        ftfy_changes += abs(len(pre_ftfy) - len(text))
        changes += ftfy_changes
        ftfy_applied = ftfy_changes > 0

    # 1. Strip invisible/zero-width characters
    invisible_count = len(_INVISIBLE_RE.findall(text))
    text = _INVISIBLE_RE.sub('', text)
    changes += invisible_count

    # 2. NFKC normalization
    pre_nfkc = text
    text = unicodedata.normalize('NFKC', text)
    nfkc_changes = sum(1 for a, b in zip(pre_nfkc, text) if a != b)
    changes += nfkc_changes

    # 3. Homoglyph folding
    pre_homoglyph = text
    text = text.translate(_HOMOGLYPH_MAP)
    homoglyph_count = sum(1 for a, b in zip(pre_homoglyph, text) if a != b)
    changes += homoglyph_count

    # 4. Inter-character spacing collapse
    interspacing_spans = len(_INTERSPACED_RE.findall(text))
    if interspacing_spans > 0:
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r'(?<!\w)(\w) (?=\w(?:\s\w)*(?!\w))', r'\1', text)
        spacing_changes = len(original) - len(text)
        changes += max(spacing_changes, 0)

    # 5. Whitespace collapse
    pre_ws = text
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    ws_collapsed = (pre_ws != text)
    if ws_collapsed:
        changes += abs(len(pre_ws) - len(text))

    # 6. Control character stripping
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    obfuscation_delta = changes / original_len

    # Classify detected evasion attack types for per-attack analysis
    attack_types = []
    if invisible_count > 0:
        attack_types.append('invisible_char')
    if homoglyph_count > 0:
        attack_types.append('homoglyph')
    if interspacing_spans > 0:
        attack_types.append('interspacing')
    if ftfy_applied:
        attack_types.append('encoding')
    if ws_collapsed:
        attack_types.append('whitespace')

    return text, {
        'obfuscation_delta': round(obfuscation_delta, 4),
        'invisible_chars': invisible_count,
        'homoglyphs': homoglyph_count,
        'interspacing_spans': interspacing_spans,
        'whitespace_collapsed': ws_collapsed,
        'ftfy_applied': ftfy_applied,
        'attack_types': attack_types,
    }
