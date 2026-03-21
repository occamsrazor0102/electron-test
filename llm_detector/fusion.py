"""Evidence fusion -- combines channel results into final determination."""

import logging

from llm_detector.channels import ChannelResult
from llm_detector.channels.prompt_structure import score_prompt_structure
from llm_detector.channels.stylometric import score_stylometric
from llm_detector.channels.continuation import score_continuation
from llm_detector.channels.windowed import score_windowed

logger = logging.getLogger(__name__)


def _detect_mode(prompt_sig, instr_density, self_sim, word_count):
    """Auto-detect whether text is a task prompt or generic AI text."""
    prompt_signal = 0.0
    if prompt_sig['composite'] >= 0.15:
        prompt_signal += prompt_sig['composite']
    if instr_density and instr_density.get('idi', 0) >= 5:
        prompt_signal += 0.3
    if prompt_sig.get('framing_completeness', 0) >= 2:
        prompt_signal += 0.2

    generic_signal = 0.0
    if self_sim and self_sim.get('nssi_signals', 0) >= 3:
        generic_signal += 0.4
    if word_count >= 400:
        generic_signal += 0.2

    if prompt_signal > generic_signal + 0.1:
        return 'task_prompt'
    elif generic_signal > prompt_signal + 0.1:
        return 'generic_aigt'
    else:
        return 'task_prompt'


def determine(preamble_score, preamble_severity, prompt_sig, voice_dis,
              instr_density=None, word_count=0,
              self_sim=None, cont_result=None, lang_gate=None, norm_report=None,
              mode='auto', fingerprint_score=0.0, semantic=None, ppl=None,
              tocsin=None, disabled_channels=None,
              semantic_flow=None, ml_fusion_enabled=False, ml_model_path=None,
              **kwargs):
    """Evidence fusion with channel-based corroboration.

    Args:
        semantic_flow: Semantic flow analyzer result dict (inter-sentence variance).
        ml_fusion_enabled: If True and a trained model exists, use ML fusion
            instead of heuristic rules. Disabled by default.
        ml_model_path: Path to trained ML fusion model (.pkl). Defaults to
            .beet/fusion_model.pkl if not specified.

    Returns (determination, reason, confidence, channel_details).
    """
    # Mode detection
    if mode == 'auto':
        mode = _detect_mode(prompt_sig, instr_density, self_sim, word_count)

    # Score all channels
    ch_prompt = score_prompt_structure(preamble_score, preamble_severity, prompt_sig, voice_dis, instr_density, word_count)
    ch_style = score_stylometric(fingerprint_score, self_sim, voice_dis, semantic=semantic, ppl=ppl, tocsin=tocsin, semantic_flow=semantic_flow)
    ch_cont = score_continuation(cont_result)
    ch_window = score_windowed(window_result=kwargs.get('window_result'))

    channels = [ch_prompt, ch_style, ch_cont, ch_window]

    _disabled = set(disabled_channels or [])

    channel_details = {
        'mode': mode,
        'disabled_channels': sorted(_disabled) if _disabled else [],
        'channels': {ch.channel: {
            'score': ch.score, 'severity': ch.severity,
            'explanation': f'{ch.channel} disabled (ablation)' if ch.channel in _disabled else ch.explanation,
            'mode_eligible': mode in ch.mode_eligibility,
            'data_sufficient': ch.data_sufficient,
            'disabled': ch.channel in _disabled,
        } for ch in channels},
    }

    # ML Fusion: if enabled and model exists, use trained classifier
    if ml_fusion_enabled:
        try:
            from llm_detector.ml_fusion import ml_determine, extract_fusion_features
            feat_names, feat_values = extract_fusion_features(channel_details, kwargs)
            ml_result = ml_determine(feat_names, feat_values, model_path=ml_model_path)
            if ml_result is not None:
                ml_det, ml_conf, ml_explanation = ml_result
                channel_details['triggering_rule'] = 'ml_fusion'
                channel_details['ml_fusion'] = {
                    'determination': ml_det,
                    'confidence': ml_conf,
                    'explanation': ml_explanation,
                }
                return ml_det, ml_explanation, ml_conf, channel_details
        except Exception as exc:
            logger.debug("ML fusion unavailable, falling back to heuristic rules: %s", exc)

    # Channel ablation: remove disabled channels from fusion
    if _disabled:
        channels = [ch for ch in channels if ch.channel not in _disabled]

    # Fairness severity cap
    severity_cap = None
    if lang_gate and lang_gate.get('support_level') == 'UNSUPPORTED':
        severity_cap = 'YELLOW'
    elif lang_gate and lang_gate.get('support_level') == 'REVIEW':
        severity_cap = 'AMBER'

    def _apply_cap(det, reason, conf):
        if severity_cap is None:
            return det, reason, conf
        sev_order = {'GREEN': 0, 'YELLOW': 1, 'REVIEW': 1, 'AMBER': 2, 'RED': 3}
        if sev_order.get(det, 0) > sev_order.get(severity_cap, 3):
            gate_reason = lang_gate.get('reason', 'language support gate')
            return severity_cap, f"{reason} [capped from {det}: {gate_reason}]", min(conf, 0.40)
        return det, reason, conf

    # L0 CRITICAL: instant RED
    if ch_prompt.sub_signals.get('preamble') == 0.99 and preamble_severity == 'CRITICAL':
        det, reason, conf = _apply_cap('RED', ch_prompt.explanation, 0.99)
        channel_details['triggering_rule'] = 'L0_critical_preamble'
        channel_details['fusion_counts'] = {
            'n_primary_red': 1, 'n_primary_amber': 0, 'n_primary_yellow_plus': 1,
            'n_yellow_plus': 1, 'n_red': 1, 'n_amber_plus': 0,
        }
        for ch in channels:
            channel_details['channels'][ch.channel]['role'] = 'primary'
        return det, reason, conf, channel_details

    # Mode-aware channel filtering
    if mode == 'task_prompt':
        primary_channels = [ch for ch in channels if 'task_prompt' in ch.mode_eligibility]
        supporting_channels = [ch for ch in channels if 'task_prompt' not in ch.mode_eligibility]
    else:
        primary_channels = channels
        supporting_channels = []

    # Evidence fusion — only count channels with sufficient data for convergence
    all_active = sorted(
        [ch for ch in channels if ch.severity != 'GREEN' and ch.data_sufficient],
        key=lambda c: c.sev_level, reverse=True,
    )
    primary_active = sorted(
        [ch for ch in primary_channels if ch.severity != 'GREEN' and ch.data_sufficient],
        key=lambda c: c.sev_level, reverse=True,
    )
    support_active = [ch for ch in supporting_channels if ch.severity != 'GREEN' and ch.data_sufficient]

    n_red = sum(1 for ch in all_active if ch.severity == 'RED')
    n_amber_plus = sum(1 for ch in all_active if ch.sev_level >= 2)
    n_yellow_plus = sum(1 for ch in all_active if ch.sev_level >= 1)
    n_primary_red = sum(1 for ch in primary_active if ch.severity == 'RED')
    n_primary_amber = sum(1 for ch in primary_active if ch.sev_level >= 2)
    n_primary_yellow_plus = sum(1 for ch in primary_active if ch.sev_level >= 1)

    top_explanations = [ch.explanation for ch in all_active[:3]]
    combined_reason = ' + '.join(top_explanations) if top_explanations else 'No significant signals'
    top_score = max((ch.score for ch in all_active), default=0.0)

    # Short-text active channel accounting
    n_active_channels = sum(1 for ch in channels if ch.score > 0 or ch.severity != 'GREEN')
    short_text = word_count > 0 and word_count < 100
    short_text_penalty = 0.15 if (short_text and n_active_channels <= 2) else 0.0
    channel_details['active_channels'] = n_active_channels
    channel_details['short_text_adjustment'] = bool(short_text_penalty)

    # Expose per-channel role (primary vs supporting) and fusion counts for reporting transparency
    primary_channel_names = {ch.channel for ch in primary_channels}
    for ch in channels:
        channel_details['channels'][ch.channel]['role'] = (
            'primary' if ch.channel in primary_channel_names else 'supporting'
        )
    channel_details['fusion_counts'] = {
        'n_primary_red': n_primary_red,
        'n_primary_amber': n_primary_amber,
        'n_primary_yellow_plus': n_primary_yellow_plus,
        'n_yellow_plus': n_yellow_plus,
        'n_red': n_red,
        'n_amber_plus': n_amber_plus,
    }

    # RED: strong primary + supporting, or two AMBER+ channels
    if n_primary_red >= 1 and n_yellow_plus >= 2:
        det, reason, conf = _apply_cap('RED', combined_reason, top_score)
        channel_details['triggering_rule'] = 'primary_red_with_corroboration'
        return det, reason, conf, channel_details

    # Short-text relaxation: 1 RED + 1 yellow is enough when few channels can run
    if short_text_penalty and n_primary_red >= 1 and n_yellow_plus >= 1:
        det, reason, conf = _apply_cap(
            'RED', f"{combined_reason} [short-text relaxed]",
            min(top_score - short_text_penalty, 0.75))
        channel_details['triggering_rule'] = 'primary_red_short_text_relaxed'
        return det, reason, conf, channel_details

    if n_primary_amber >= 2:
        det, reason, conf = _apply_cap('RED', combined_reason, min(top_score, 0.85))
        channel_details['triggering_rule'] = 'two_primary_amber_channels'
        return det, reason, conf, channel_details

    if mode == 'task_prompt' and n_primary_red >= 1 and n_yellow_plus == 1:
        det, reason, conf = _apply_cap('AMBER', f"{combined_reason} [single-channel, demoted from RED]", min(top_score, 0.75))
        channel_details['triggering_rule'] = 'primary_red_single_channel_demoted'
        return det, reason, conf, channel_details

    if mode == 'generic_aigt' and n_red >= 1:
        if n_yellow_plus >= 2:
            det, reason, conf = _apply_cap('RED', combined_reason, top_score)
            channel_details['triggering_rule'] = 'generic_aigt_red_with_corroboration'
        else:
            det, reason, conf = _apply_cap('RED', f"{combined_reason} [single-channel]", min(top_score, 0.75))
            channel_details['triggering_rule'] = 'generic_aigt_red_single_channel'
        return det, reason, conf, channel_details

    # AMBER: one channel at AMBER, or two at YELLOW+
    if n_primary_amber >= 1:
        det, reason, conf = _apply_cap('AMBER', combined_reason, min(top_score, 0.70))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            channel_details['triggering_rule'] = 'primary_amber_mixed_windowing'
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.60), channel_details
        channel_details['triggering_rule'] = 'primary_amber_single_channel'
        return det, reason, conf, channel_details

    if mode == 'task_prompt':
        convergence_count = n_primary_yellow_plus + min(1, len(support_active))
    else:
        convergence_count = n_yellow_plus

    if convergence_count >= 2:
        det, reason, conf = _apply_cap('AMBER', f"{combined_reason} [multi-channel convergence]", min(top_score, 0.60))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            channel_details['triggering_rule'] = 'multi_channel_convergence_mixed'
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.55), channel_details
        channel_details['triggering_rule'] = 'multi_channel_convergence'
        return det, reason, conf, channel_details

    # Supporting channels at AMBER in task_prompt mode
    if mode == 'task_prompt' and any(ch.sev_level >= 2 for ch in support_active):
        support_expl = [ch.explanation for ch in support_active if ch.sev_level >= 2]
        det, reason, conf = _apply_cap('AMBER', f"{' + '.join(support_expl)} [supporting channel]", 0.55)
        channel_details['triggering_rule'] = 'supporting_channel_amber'
        return det, reason, conf, channel_details

    # YELLOW: one channel at YELLOW+
    if n_yellow_plus >= 1:
        det, reason, conf = _apply_cap('YELLOW', combined_reason, min(top_score, 0.45))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            channel_details['triggering_rule'] = 'yellow_signal_mixed_windowing'
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.50), channel_details
        channel_details['triggering_rule'] = 'yellow_signal'
        return det, reason, conf, channel_details

    # Obfuscation delta
    if norm_report and norm_report.get('obfuscation_delta', 0) >= 0.05:
        delta = norm_report['obfuscation_delta']
        det, reason, conf = _apply_cap('YELLOW', f"Text normalization delta ({delta:.1%}) suggests obfuscation", 0.35)
        channel_details['triggering_rule'] = 'obfuscation_delta'
        return det, reason, conf, channel_details

    # REVIEW: any channel has non-zero score
    any_signal = any(ch.score > 0.05 for ch in channels)
    if any_signal:
        weak_parts = [ch.explanation for ch in channels if ch.score > 0.05]
        channel_details['triggering_rule'] = 'weak_signals_below_threshold'
        return 'REVIEW', f"Weak signals below threshold: {' + '.join(weak_parts[:2])}", 0.10, channel_details

    # GREEN
    channel_details['triggering_rule'] = 'no_signal'
    return 'GREEN', 'No significant signals', 0.0, channel_details
