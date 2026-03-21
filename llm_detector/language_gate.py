"""
Fairness / Language support gate.

Caps severity if text is outside the validated English-prose envelope.
Ref: Liang et al. (2023) "GPT Detectors Are Biased Against Non-Native
     English Writers"
Ref: Wang et al. (2023) "M4 -- multilingual detection remains harder."
"""

import unicodedata
from llm_detector.text_utils import ENGLISH_FUNCTION_WORDS


def check_language_support(text, word_count=None):
    """Assess whether text is within the validated English-prose envelope.

    Returns dict with support_level, function_word_coverage, non_latin_ratio, reason.
    """
    words = text.lower().split()
    if word_count is None:
        word_count = len(words)

    if word_count < 30:
        return {
            'support_level': 'REVIEW',
            'function_word_coverage': 0.0,
            'non_latin_ratio': 0.0,
            'reason': 'Text too short for reliable English detection',
        }

    fw_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    fw_coverage = fw_count / max(word_count, 1)

    alpha_chars = [c for c in text if c.isalpha()]
    n_alpha = max(len(alpha_chars), 1)
    non_latin = sum(1 for c in alpha_chars
                    if unicodedata.category(c).startswith('L')
                    and not ('\u0041' <= c <= '\u007a' or '\u00c0' <= c <= '\u024f'))
    non_latin_ratio = non_latin / n_alpha

    if non_latin_ratio > 0.30:
        level = 'UNSUPPORTED'
        reason = f'High non-Latin script content ({non_latin_ratio:.0%})'
    elif fw_coverage < 0.08:
        level = 'UNSUPPORTED'
        reason = f'Very low English function-word coverage ({fw_coverage:.0%})'
    elif fw_coverage < 0.12:
        level = 'REVIEW'
        reason = f'Low English function-word coverage ({fw_coverage:.0%}) -- possible non-native or non-English text'
    elif non_latin_ratio > 0.10:
        level = 'REVIEW'
        reason = f'Mixed-script content ({non_latin_ratio:.0%} non-Latin)'
    else:
        level = 'SUPPORTED'
        reason = 'Text within validated English-prose envelope'

    return {
        'support_level': level,
        'function_word_coverage': round(fw_coverage, 4),
        'non_latin_ratio': round(non_latin_ratio, 4),
        'reason': reason,
    }
