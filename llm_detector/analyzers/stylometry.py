"""Topic-scrubbed stylometric feature extraction.

Masks topical content before computing style features to reduce topic leakage.
"""

import re
import statistics
from collections import Counter

from llm_detector.text_utils import ENGLISH_FUNCTION_WORDS, get_sentences, type_token_ratio


def _mattr(words, window=50):
    """Moving-Average Type-Token Ratio over rolling windows.

    Eliminates document-length bias inherent in standard TTR by averaging
    the TTR across overlapping windows of fixed size.

    Args:
        words: List of word tokens (already lowercased).
        window: Window size in words (default 50).

    Returns float MATTR value.
    """
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    ratios = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return statistics.mean(ratios)

# Topic masking patterns
_TOPIC_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
_TOPIC_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_TOPIC_DATE_RE = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    r'|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*[\s,]+\d{1,2},?\s*\d{4}\b',
    re.I,
)
_TOPIC_FILENAME_RE = re.compile(r'\b\w+\.\w{2,4}\b')
_TOPIC_VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)*\b', re.I)
_TOPIC_NUMBER_RE = re.compile(r'\b\d{3,}\b')
_TOPIC_CAMELCASE_RE = re.compile(r'\b[a-z]+[A-Z]\w+\b|\b[A-Z][a-z]+[A-Z]\w*\b')
_TOPIC_ALLCAPS_RE = re.compile(r'\b[A-Z]{2,}\b')


def mask_topical_content(text):
    """Replace topical tokens with placeholders. Returns (masked_text, mask_count)."""
    count = 0
    for pattern, repl in [
        (_TOPIC_URL_RE, ' _URL_ '),
        (_TOPIC_EMAIL_RE, ' _EMAIL_ '),
        (_TOPIC_DATE_RE, ' _DATE_ '),
        (_TOPIC_FILENAME_RE, ' _FILE_ '),
        (_TOPIC_VERSION_RE, ' _VER_ '),
        (_TOPIC_NUMBER_RE, ' _NUM_ '),
        (_TOPIC_CAMELCASE_RE, ' _IDENT_ '),
        (_TOPIC_ALLCAPS_RE, ' _ACRO_ '),
    ]:
        text, n = pattern.subn(repl, text)
        count += n
    return text, count


def extract_stylometric_features(text, masked_text=None):
    """Extract topic-invariant stylometric features.

    Returns dict with char n-gram profile, function word ratio, punctuation
    bigrams, sentence length dispersion, type-token ratio, etc.
    """
    if masked_text is None:
        masked_text, _ = mask_topical_content(text)

    words = masked_text.lower().split()
    n_words = max(len(words), 1)

    # Character 4-grams
    lower_masked = masked_text.lower()
    char4 = Counter()
    for i in range(len(lower_masked) - 3):
        gram = lower_masked[i:i+4]
        if not gram.startswith('_'):
            char4[gram] += 1
    total_4grams = max(sum(char4.values()), 1)
    char_ngram_profile = {g: c / total_4grams for g, c in char4.most_common(50)}

    # Function word ratio
    fw_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    function_word_ratio = fw_count / n_words

    # Punctuation bigrams
    punct_chars = re.findall(r'[^\w\s]', text)
    punct_bigrams = Counter()
    for i in range(len(punct_chars) - 1):
        punct_bigrams[punct_chars[i] + punct_chars[i+1]] += 1

    # Sentence length dispersion
    sentences = get_sentences(text)
    sent_lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(sent_lengths) >= 2:
        mean_sl = statistics.mean(sent_lengths)
        std_sl = statistics.stdev(sent_lengths)
        sent_length_dispersion = std_sl / max(mean_sl, 1)
    else:
        sent_length_dispersion = 0.0

    # Type-token ratio
    orig_words = re.findall(r'\w+', text.lower())
    n_orig = max(len(orig_words), 1)
    type_token_ratio_val = type_token_ratio(orig_words)

    # Average word length
    word_lengths = [len(w) for w in orig_words]
    avg_word_length = statistics.mean(word_lengths) if word_lengths else 0

    # Short word ratio
    short_words = sum(1 for w in orig_words if len(w) <= 3)
    short_word_ratio = short_words / n_orig

    # MATTR: Moving-Average Type-Token Ratio (length-independent lexical diversity)
    mattr_val = _mattr(orig_words, window=50)

    return {
        'char_ngram_profile': char_ngram_profile,
        'function_word_ratio': round(function_word_ratio, 4),
        'punct_bigrams': dict(punct_bigrams.most_common(20)),
        'sent_length_dispersion': round(sent_length_dispersion, 4),
        'type_token_ratio': round(type_token_ratio_val, 4),
        'avg_word_length': round(avg_word_length, 2),
        'short_word_ratio': round(short_word_ratio, 4),
        'mattr': round(mattr_val, 4),
    }
