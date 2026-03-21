"""Channel 4: Sentence-window scoring for mixed content detection."""

from llm_detector.channels import ChannelResult


def score_windowed(window_result=None):
    """Score windowed channel. Returns ChannelResult."""
    if window_result is None or window_result.get('n_windows', 0) == 0:
        return ChannelResult(
            'windowing', 0.0, 'GREEN',
            'Windowing: insufficient text for windows',
            mode_eligibility=['generic_aigt'],
            sub_signals={},
            data_sufficient=False,
        )

    sub = {
        'max_window': window_result['max_window_score'],
        'mean_window': window_result['mean_window_score'],
        'variance': window_result['window_variance'],
        'hot_span': window_result['hot_span_length'],
        'n_windows': window_result['n_windows'],
        'mixed_signal': window_result['mixed_signal'],
    }

    score = 0.0
    severity = 'GREEN'
    parts = []

    max_w = window_result['max_window_score']
    hot_span = window_result['hot_span_length']
    variance = window_result['window_variance']
    mixed = window_result['mixed_signal']

    if max_w >= 0.60 and hot_span >= 3:
        score = max(score, 0.75)
        severity = 'RED'
        parts.append(f"hot_span={hot_span}(max={max_w:.2f})")
    elif max_w >= 0.45 and hot_span >= 2:
        score = max(score, 0.55)
        severity = 'AMBER'
        parts.append(f"hot_span={hot_span}(max={max_w:.2f})")
    elif max_w >= 0.30:
        score = max(score, 0.30)
        severity = 'YELLOW'
        parts.append(f"max_window={max_w:.2f}")

    if mixed and severity != 'GREEN':
        parts.append(f"MIXED(var={variance:.3f})")

    explanation = f"Windowing: {', '.join(parts)}" if parts else 'Windowing: no signals'

    return ChannelResult(
        'windowing', score, severity, explanation,
        mode_eligibility=['generic_aigt'],
        sub_signals=sub,
    )
