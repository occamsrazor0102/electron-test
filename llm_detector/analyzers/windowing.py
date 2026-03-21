"""Windowed scoring -- detect mixed human+AI content via per-window analysis.

Ref: M4GT-Bench (Wang et al. 2024) -- mixed detection as separate task.
"""

import re
import zlib
import statistics

from llm_detector.text_utils import ENGLISH_FUNCTION_WORDS, get_sentences, get_sentence_spans
from llm_detector.analyzers.self_similarity import _FORMULAIC_PATTERNS, _TRANSITION, _POWER_ADJ


def detect_changepoint(feature_sequence, threshold=3.0):
    """CUSUM changepoint detection on a 1D feature sequence.

    Returns dict with changepoint index and effect size, or None.
    """
    if len(feature_sequence) < 6:
        return None

    n = len(feature_sequence)
    mean_all = statistics.mean(feature_sequence)

    cusum = [0.0]
    for val in feature_sequence:
        cusum.append(cusum[-1] + (val - mean_all))

    max_dev = 0.0
    best_idx = None
    for i in range(1, n):
        dev = abs(cusum[i])
        if dev > max_dev:
            max_dev = dev
            best_idx = i

    if best_idx is None or best_idx < 2 or best_idx > n - 2:
        return None

    before = feature_sequence[:best_idx]
    after = feature_sequence[best_idx:]
    if len(before) < 2 or len(after) < 2:
        return None

    mean_before = statistics.mean(before)
    mean_after = statistics.mean(after)
    pooled_std = statistics.stdev(feature_sequence)

    if pooled_std < 1e-6:
        return None

    effect_size = abs(mean_after - mean_before) / pooled_std
    if effect_size < threshold:
        return None

    return {
        'changepoint_sentence': best_idx,
        'effect_size': round(effect_size, 3),
        'mean_before': round(mean_before, 4),
        'mean_after': round(mean_after, 4),
    }


def score_surprisal_windows(token_losses, window_size=64, stride=32):
    """Score text using windowed token-level surprisal statistics.

    Args:
        token_losses: 1D list of per-token loss values (from DivEye/perplexity).
        window_size: Tokens per window.
        stride: Token stride between windows.

    Returns dict with surprisal trajectory features.
    """
    if len(token_losses) < window_size:
        return {
            'surprisal_windows': 0,
            'surprisal_trajectory_cv': 0.0,
            'surprisal_var_of_var': 0.0,
            'surprisal_stationarity': 0.0,
        }

    window_means = []
    window_vars = []

    for start in range(0, len(token_losses) - window_size + 1, stride):
        chunk = token_losses[start:start + window_size]
        if hasattr(chunk, 'mean'):  # torch tensor
            window_means.append(chunk.mean().item())
            window_vars.append(chunk.std().item() if len(chunk) > 1 else 0.0)
        else:
            window_means.append(statistics.mean(chunk))
            window_vars.append(statistics.stdev(chunk) if len(chunk) > 1 else 0.0)

    if len(window_means) < 3:
        return {
            'surprisal_windows': len(window_means),
            'surprisal_trajectory_cv': 0.0,
            'surprisal_var_of_var': 0.0,
            'surprisal_stationarity': 0.0,
        }

    trajectory_cv = statistics.stdev(window_means) / max(statistics.mean(window_means), 1e-6)
    var_of_var = statistics.stdev(window_vars) / max(statistics.mean(window_vars), 1e-6)

    stationarity_score = max(0.0, 1.0 - trajectory_cv) * max(0.0, 1.0 - var_of_var)

    return {
        'surprisal_windows': len(window_means),
        'surprisal_trajectory_cv': round(trajectory_cv, 4),
        'surprisal_var_of_var': round(var_of_var, 4),
        'surprisal_stationarity': round(stationarity_score, 4),
    }


def score_windows(text, window_size=5, stride=2):
    """Score text in overlapping sentence windows.

    Returns dict with per-window scores, max/mean/variance, hot span, mixed signal.
    """
    sentences = get_sentences(text)
    if len(sentences) < window_size:
        return {
            'windows': [],
            'max_window_score': 0.0,
            'mean_window_score': 0.0,
            'window_variance': 0.0,
            'hot_span_length': 0,
            'n_windows': 0,
            'mixed_signal': False,
            'fw_trajectory_cv': 0.0,
            'comp_trajectory_mean': 0.0,
            'comp_trajectory_cv': 0.0,
            'changepoint': None,
        }

    windows = []
    fw_ratios = []
    comp_ratios = []
    for start in range(0, len(sentences) - window_size + 1, stride):
        end = start + window_size
        window_text = ' '.join(sentences[start:end])
        window_words = window_text.split()
        n_w = max(len(window_words), 1)

        formulaic_count = sum(
            len(compiled_pat.findall(window_text))
            for compiled_pat, _weight in _FORMULAIC_PATTERNS
        )
        formulaic_density = formulaic_count / (n_w / 100)

        trans_hits = len(_TRANSITION.findall(window_text))
        trans_density = trans_hits / (n_w / 100)

        power_hits = len(_POWER_ADJ.findall(window_text))
        power_density = power_hits / (n_w / 100)

        fw = sum(1 for w in window_words if w.lower() in ENGLISH_FUNCTION_WORDS)
        fw_ratio = fw / n_w
        fw_ratios.append(fw_ratio)

        # FEAT 4: Per-window compression
        window_bytes = window_text.encode('utf-8')
        if len(window_bytes) > 20:
            window_comp = len(zlib.compress(window_bytes)) / len(window_bytes)
        else:
            window_comp = 0.5
        comp_ratios.append(window_comp)

        w_sent_lengths = [len(s.split()) for s in sentences[start:end] if s.strip()]
        if len(w_sent_lengths) >= 2:
            w_mean = statistics.mean(w_sent_lengths)
            w_std = statistics.stdev(w_sent_lengths)
            w_cv = w_std / max(w_mean, 1)
        else:
            w_cv = 0.5

        ai_indicators = 0.0
        if formulaic_density > 2.0:
            ai_indicators += min(formulaic_density / 5.0, 0.3)
        if trans_density > 3.0:
            ai_indicators += min(trans_density / 8.0, 0.2)
        if power_density > 1.5:
            ai_indicators += min(power_density / 4.0, 0.2)
        if w_cv < 0.25 and len(w_sent_lengths) >= 3:
            ai_indicators += 0.15
        if fw_ratio < 0.12:
            ai_indicators += 0.15

        window_score = min(ai_indicators, 1.0)

        windows.append({
            'start': start,
            'end': end,
            'score': round(window_score, 3),
            'formulaic': round(formulaic_density, 2),
            'transitions': round(trans_density, 2),
            'sent_cv': round(w_cv, 3),
        })

    scores = [w['score'] for w in windows]
    max_score = max(scores) if scores else 0.0
    mean_score = statistics.mean(scores) if scores else 0.0
    variance = statistics.variance(scores) if len(scores) >= 2 else 0.0

    hot_threshold = 0.30
    hot_span = 0
    current_span = 0
    for s in scores:
        if s >= hot_threshold:
            current_span += 1
            hot_span = max(hot_span, current_span)
        else:
            current_span = 0

    mixed_signal = variance >= 0.02 and max_score >= 0.30 and mean_score < 0.50

    # FEAT 3: Function word trajectory CV
    if len(fw_ratios) >= 3:
        fw_trajectory_cv = statistics.stdev(fw_ratios) / max(statistics.mean(fw_ratios), 0.01)
    else:
        fw_trajectory_cv = 0.0

    # FEAT 4: Compression trajectory
    if len(comp_ratios) >= 3:
        comp_trajectory_cv = statistics.stdev(comp_ratios) / max(statistics.mean(comp_ratios), 0.01)
        comp_trajectory_mean = statistics.mean(comp_ratios)
    else:
        comp_trajectory_cv = 0.0
        comp_trajectory_mean = 0.0

    # FEAT 9: Changepoint detection
    changepoint = None
    if len(scores) >= 6:
        changepoint = detect_changepoint(scores)
        if changepoint:
            changepoint['changepoint_sentence'] = changepoint['changepoint_sentence'] * stride

    return {
        'windows': windows,
        'max_window_score': round(max_score, 3),
        'mean_window_score': round(mean_score, 3),
        'window_variance': round(variance, 4),
        'hot_span_length': hot_span,
        'n_windows': len(windows),
        'mixed_signal': mixed_signal,
        'fw_trajectory_cv': round(fw_trajectory_cv, 4),
        'comp_trajectory_mean': round(comp_trajectory_mean, 4),
        'comp_trajectory_cv': round(comp_trajectory_cv, 4),
        'changepoint': changepoint,
    }


def get_hot_window_spans(text, threshold=0.30, window_size=5, stride=2,
                         precomputed_result=None):
    """Return character-level spans for hot (high-scoring) sentence windows.

    Args:
        text: Original text.
        threshold: Minimum window score to include (default 0.30).
        window_size: Sentences per window.
        stride: Sentence stride.
        precomputed_result: Optional dict returned by a prior ``score_windows``
            call for the same text/parameters.  When provided the function
            skips the redundant ``score_windows`` computation.

    Returns list of (start_char, end_char, score, 'hot_window', window_index).
    """
    sent_spans = get_sentence_spans(text)
    if len(sent_spans) < window_size:
        return []

    result = (precomputed_result if precomputed_result is not None
              else score_windows(text, window_size=window_size, stride=stride))
    spans = []

    for w in result['windows']:
        if w['score'] >= threshold:
            w_start = w['start']
            w_end = min(w['end'] - 1, len(sent_spans) - 1)
            if w_start < len(sent_spans) and w_end < len(sent_spans):
                char_start = sent_spans[w_start][1]
                char_end = sent_spans[w_end][2]
                spans.append((char_start, char_end, w['score'], 'hot_window', w_start))

    return spans
