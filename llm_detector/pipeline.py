"""Full analysis pipeline orchestration."""

from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
from llm_detector.normalize import normalize_text
from llm_detector.language_gate import check_language_support
from llm_detector.analyzers.preamble import run_preamble
from llm_detector.analyzers.fingerprint import run_fingerprint, run_fingerprint_spans
from llm_detector.lexicon.integration import (
    run_prompt_signature_enhanced,
    run_voice_dissonance_enhanced,
    run_instruction_density_enhanced,
)
from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
from llm_detector.analyzers.self_similarity import run_self_similarity
from llm_detector.analyzers.continuation_api import run_continuation_api_multi
from llm_detector.analyzers.continuation_local import run_continuation_local_multi
from llm_detector.analyzers.perplexity import run_perplexity
from llm_detector.analyzers.token_cohesiveness import run_token_cohesiveness
from llm_detector.analyzers.stylometry import mask_topical_content, extract_stylometric_features
from llm_detector.analyzers.semantic_flow import run_semantic_flow
from llm_detector.analyzers.windowing import score_windows, score_surprisal_windows, get_hot_window_spans
from llm_detector.fusion import determine
from llm_detector.calibration import apply_calibration


def analyze_prompt(text, task_id='', occupation='', attempter='', stage='',
                   run_l3=True, api_key=None, dna_provider='anthropic',
                   dna_model=None, dna_samples=3,
                   ground_truth=None, language=None, domain=None,
                   mode='auto', cal_table=None, memory_store=None,
                   disabled_channels=None, precomputed_continuation=None,
                   ppl_model=None):
    """Run full v0.68 pipeline on a single prompt. Returns result dict."""
    # Normalization pre-pass
    normalized_text, norm_report = normalize_text(text)
    word_count_raw = len(text.split())
    word_count = len(normalized_text.split())

    # Fairness / language support gate
    lang_gate = check_language_support(normalized_text, word_count)

    text_for_analysis = normalized_text

    # Run all analyzers
    preamble_score, preamble_severity, preamble_hits, preamble_spans = run_preamble(text_for_analysis)
    fingerprint_score, fingerprint_hits, fingerprint_rate = run_fingerprint(text_for_analysis)
    prompt_sig = run_prompt_signature_enhanced(text_for_analysis)
    voice_dis = run_voice_dissonance_enhanced(text_for_analysis)
    instr_density = run_instruction_density_enhanced(
        text_for_analysis,
        constraint_active=(prompt_sig.get('pack_constraint_score', 0) > 0.08),
        schema_active=(voice_dis.get('pack_schema_score', 0) > 0.05),
    )

    self_sim = None
    if run_l3:
        self_sim = run_self_similarity(text_for_analysis)

    cont_result = None
    if precomputed_continuation is not None:
        cont_result = precomputed_continuation
    elif run_l3 and api_key:
        cont_result = run_continuation_api_multi(
            text_for_analysis, api_key=api_key, provider=dna_provider,
            model=dna_model, n_samples=dna_samples,
        )
    elif run_l3:
        cont_result = run_continuation_local_multi(text_for_analysis)

    semantic = run_semantic_resonance(text_for_analysis)
    ppl = run_perplexity(text_for_analysis, model_id=ppl_model)
    tocsin = run_token_cohesiveness(text_for_analysis)

    # Semantic flow: inter-sentence embedding similarity variance
    semantic_flow = run_semantic_flow(text_for_analysis)

    # FEAT 10: Surprisal trajectory from per-token losses
    surprisal_traj = {}
    token_losses = ppl.get('token_losses')
    if token_losses:
        surprisal_traj = score_surprisal_windows(token_losses)

    # Topic-scrubbed stylometry
    masked_text, mask_count = mask_topical_content(text_for_analysis)
    stylo_features = extract_stylometric_features(text_for_analysis, masked_text)

    # Windowed scoring
    window_result = score_windows(text_for_analysis)

    # Detection spans — merged from all annotation sources
    detection_spans = list(preamble_spans)
    detection_spans.extend(
        {'start': s, 'end': e, 'text': t, 'source': 'fingerprint', 'label': w, 'type': 'fingerprint'}
        for s, e, t, _, w in run_fingerprint_spans(text_for_analysis)
    )
    detection_spans.extend(prompt_sig.get('pack_spans', []))
    detection_spans.extend(voice_dis.get('pack_spans', []))
    detection_spans.extend(instr_density.get('pack_spans', []))
    for hw in get_hot_window_spans(text_for_analysis, precomputed_result=window_result):
        detection_spans.append({
            'start': hw[0], 'end': hw[1], 'text': '', 'source': 'hot_window',
            'label': f'score={hw[2]:.2f}', 'type': 'window',
        })
    detection_spans.sort(key=lambda x: x.get('start', 0))

    # Evidence fusion
    det, reason, confidence, channel_details = determine(
        preamble_score, preamble_severity, prompt_sig, voice_dis, instr_density, word_count,
        self_sim=self_sim, cont_result=cont_result,
        lang_gate=lang_gate, norm_report=norm_report,
        mode=mode, fingerprint_score=fingerprint_score,
        semantic=semantic, ppl=ppl,
        tocsin=tocsin,
        semantic_flow=semantic_flow,
        window_result=window_result,
        disabled_channels=disabled_channels,
    )

    # Conformal calibration
    if word_count < 100:
        length_bin = 'short'
    elif word_count < 300:
        length_bin = 'medium'
    elif word_count < 800:
        length_bin = 'long'
    else:
        length_bin = 'very_long'

    cal_result = apply_calibration(confidence, cal_table, domain=domain, length_bin=length_bin)

    # Audit trail
    audit_trail = {
        'pipeline_version': 'v0.68',
        'mode_resolved': channel_details.get('mode', mode),
        'channels': channel_details.get('channels', {}),
        'fairness_gate': {
            'support_level': lang_gate.get('support_level'),
            'fw_coverage': lang_gate.get('function_word_coverage'),
        },
        'normalization': {
            'obfuscation_delta': norm_report.get('obfuscation_delta', 0),
            'invisible_chars': norm_report.get('invisible_chars', 0),
            'homoglyphs': norm_report.get('homoglyphs', 0),
            'ftfy_applied': norm_report.get('ftfy_applied', False),
        },
        'calibration': cal_result,
        'semantic_available': HAS_SEMANTIC,
        'perplexity_available': HAS_PERPLEXITY,
    }

    result = {
        'task_id': task_id,
        'occupation': occupation,
        'attempter': attempter,
        'stage': stage,
        'word_count': word_count,
        'word_count_raw': word_count_raw,
        'determination': det,
        'reason': reason,
        'confidence': confidence,
        'calibrated_confidence': cal_result['calibrated_confidence'],
        'conformity_level': cal_result['conformity_level'],
        'calibration_stratum': cal_result['stratum_used'],
        'mode': channel_details.get('mode', mode),
        'channel_details': channel_details,
        'audit_trail': audit_trail,
        'pipeline_version': 'v0.68',
        # Detection spans
        'detection_spans': detection_spans,
        # Normalization
        'norm_obfuscation_delta': norm_report.get('obfuscation_delta', 0.0),
        'norm_invisible_chars': norm_report.get('invisible_chars', 0),
        'norm_homoglyphs': norm_report.get('homoglyphs', 0),
        'norm_attack_types': norm_report.get('attack_types', []),
        # Fairness gate
        'lang_support_level': lang_gate.get('support_level', 'SUPPORTED'),
        'lang_fw_coverage': lang_gate.get('function_word_coverage', 0.0),
        'lang_non_latin_ratio': lang_gate.get('non_latin_ratio', 0.0),
        # Preamble
        'preamble_score': preamble_score,
        'preamble_severity': preamble_severity,
        'preamble_hits': len(preamble_hits),
        'preamble_details': preamble_hits,
        # Fingerprint (diagnostic-only)
        'fingerprint_score': fingerprint_score,
        'fingerprint_hits': fingerprint_hits,
        # Prompt signature
        'prompt_signature_composite': prompt_sig['composite'],
        'prompt_signature_cfd': prompt_sig['cfd'],
        'prompt_signature_distinct_frames': prompt_sig['distinct_frames'],
        'prompt_signature_mfsr': prompt_sig['mfsr'],
        'prompt_signature_framing': prompt_sig['framing_completeness'],
        'prompt_signature_conditional_density': prompt_sig['conditional_density'],
        'prompt_signature_meta_design': prompt_sig['meta_design_hits'],
        'prompt_signature_contractions': prompt_sig['contractions'],
        'prompt_signature_must_rate': prompt_sig['must_rate'],
        'prompt_signature_numbered_criteria': prompt_sig['numbered_criteria'],
        # Instruction density
        'instruction_density_idi': instr_density['idi'],
        'instruction_density_imperatives': instr_density['imperatives'],
        'instruction_density_conditionals': instr_density['conditionals'],
        'instruction_density_binary_specs': instr_density['binary_specs'],
        'instruction_density_missing_refs': instr_density['missing_refs'],
        'instruction_density_flag_count': instr_density['flag_count'],
        # Voice dissonance
        'voice_dissonance_voice_score': voice_dis['voice_score'],
        'voice_dissonance_spec_score': voice_dis['spec_score'],
        'voice_dissonance_vsd': voice_dis['vsd'],
        'voice_dissonance_voice_gated': voice_dis['voice_gated'],
        'voice_dissonance_casual_markers': voice_dis['casual_markers'],
        'voice_dissonance_misspellings': voice_dis['misspellings'],
        'voice_dissonance_camel_cols': voice_dis['camel_cols'],
        'voice_dissonance_calcs': voice_dis['calcs'],
        'voice_dissonance_hedges': voice_dis['hedges'],
        # SSI
        'ssi_triggered': (
            voice_dis['spec_score'] >= (5.0 if voice_dis['contractions'] == 0 else 7.0)
            and voice_dis['voice_score'] < 0.5
            and voice_dis['hedges'] == 0
            and word_count >= 150
        ),
        # Metadata
        'ground_truth': ground_truth,
        'language': language,
        'domain': domain,
        # Windowed scoring
        'window_max_score': window_result.get('max_window_score', 0.0),
        'window_mean_score': window_result.get('mean_window_score', 0.0),
        'window_variance': window_result.get('window_variance', 0.0),
        'window_hot_span': window_result.get('hot_span_length', 0),
        'window_n_windows': window_result.get('n_windows', 0),
        'window_mixed_signal': window_result.get('mixed_signal', False),
        'window_fw_trajectory_cv': window_result.get('fw_trajectory_cv', 0.0),
        'window_comp_trajectory_mean': window_result.get('comp_trajectory_mean', 0.0),
        'window_comp_trajectory_cv': window_result.get('comp_trajectory_cv', 0.0),
        'window_changepoint': window_result.get('changepoint'),
        # Pack diagnostics
        'pack_constraint_score': prompt_sig.get('pack_constraint_score', 0.0),
        'pack_exec_spec_score': prompt_sig.get('pack_exec_spec_score', 0.0),
        'pack_schema_score': voice_dis.get('pack_schema_score', 0.0),
        'pack_active_families': prompt_sig.get('pack_active_families', 0),
        'pack_prompt_boost': prompt_sig.get('pack_boost', 0.0),
        'pack_idi_boost': instr_density.get('pack_idi_boost', 0.0),
        # Stylometric features
        'stylo_fw_ratio': stylo_features.get('function_word_ratio', 0.0),
        'stylo_sent_dispersion': stylo_features.get('sent_length_dispersion', 0.0),
        'stylo_ttr': stylo_features.get('type_token_ratio', 0.0),
        'stylo_avg_word_len': stylo_features.get('avg_word_length', 0.0),
        'stylo_short_word_ratio': stylo_features.get('short_word_ratio', 0.0),
        'stylo_mask_count': mask_count,
        'stylo_mattr': stylo_features.get('mattr', 0.0),
    }

    # Semantic resonance
    result.update({
        'semantic_resonance_ai_score': semantic.get('semantic_ai_score', 0.0),
        'semantic_resonance_human_score': semantic.get('semantic_human_score', 0.0),
        'semantic_resonance_ai_mean': semantic.get('semantic_ai_mean', 0.0),
        'semantic_resonance_human_mean': semantic.get('semantic_human_mean', 0.0),
        'semantic_resonance_delta': semantic.get('semantic_delta', 0.0),
        'semantic_resonance_determination': semantic.get('determination'),
        'semantic_resonance_confidence': semantic.get('confidence', 0.0),
    })

    # Perplexity
    result.update({
        'perplexity_value': ppl.get('perplexity', 0.0),
        'perplexity_determination': ppl.get('determination'),
        'perplexity_confidence': ppl.get('confidence', 0.0),
        'surprisal_variance': ppl.get('surprisal_variance', 0.0),
        'surprisal_first_half_var': ppl.get('surprisal_first_half_var', 0.0),
        'surprisal_second_half_var': ppl.get('surprisal_second_half_var', 0.0),
        'volatility_decay_ratio': ppl.get('volatility_decay_ratio', 1.0),
        'binoculars_score': ppl.get('binoculars_score', 0.0),
        'binoculars_determination': ppl.get('binoculars_determination'),
    })

    # Self-similarity (NSSI)
    if self_sim:
        result.update({
            'self_similarity_nssi_score': self_sim.get('nssi_score', 0.0),
            'self_similarity_nssi_signals': self_sim.get('nssi_signals', 0),
            'self_similarity_determination': self_sim.get('determination'),
            'self_similarity_confidence': self_sim.get('confidence', 0.0),
            'self_similarity_formulaic_density': self_sim.get('formulaic_density', 0.0),
            'self_similarity_power_adj_density': self_sim.get('power_adj_density', 0.0),
            'self_similarity_demonstrative_density': self_sim.get('demonstrative_density', 0.0),
            'self_similarity_transition_density': self_sim.get('transition_density', 0.0),
            'self_similarity_scare_quote_density': self_sim.get('scare_quote_density', 0.0),
            'self_similarity_emdash_density': self_sim.get('emdash_density', 0.0),
            'self_similarity_this_the_start_rate': self_sim.get('this_the_start_rate', 0.0),
            'self_similarity_section_depth': self_sim.get('section_depth', 0),
            'self_similarity_sent_length_cv': self_sim.get('sent_length_cv', 0.0),
            'self_similarity_comp_ratio': self_sim.get('comp_ratio', 0.0),
            'self_similarity_hapax_ratio': self_sim.get('hapax_ratio', 0.0),
            'self_similarity_hapax_count': self_sim.get('hapax_count', 0),
            'self_similarity_unique_words': self_sim.get('unique_words', 0),
            'self_similarity_shuffled_comp_ratio': self_sim.get('shuffled_comp_ratio', 0.0),
            'self_similarity_structural_compression_delta': self_sim.get('structural_compression_delta', 0.0),
        })
    else:
        result.update({
            'self_similarity_nssi_score': 0.0, 'self_similarity_nssi_signals': 0,
            'self_similarity_determination': None, 'self_similarity_confidence': 0.0,
            'self_similarity_formulaic_density': 0.0, 'self_similarity_power_adj_density': 0.0,
            'self_similarity_demonstrative_density': 0.0, 'self_similarity_transition_density': 0.0,
            'self_similarity_scare_quote_density': 0.0, 'self_similarity_emdash_density': 0.0,
            'self_similarity_this_the_start_rate': 0.0, 'self_similarity_section_depth': 0,
            'self_similarity_sent_length_cv': 0.0, 'self_similarity_comp_ratio': 0.0,
            'self_similarity_hapax_ratio': 0.0, 'self_similarity_hapax_count': 0,
            'self_similarity_unique_words': 0,
            'self_similarity_shuffled_comp_ratio': 0.0,
            'self_similarity_structural_compression_delta': 0.0,
        })

    # Continuation (DNA-GPT)
    if cont_result:
        proxy = cont_result.get('proxy_features', {})
        result.update({
            'continuation_bscore': cont_result.get('bscore', 0.0),
            'continuation_bscore_max': cont_result.get('bscore_max', 0.0),
            'continuation_determination': cont_result.get('determination'),
            'continuation_confidence': cont_result.get('confidence', 0.0),
            'continuation_n_samples': cont_result.get('n_samples', 0),
            'continuation_mode': 'local' if proxy else 'api',
            'continuation_ncd': proxy.get('ncd', 0.0),
            'continuation_internal_overlap': proxy.get('internal_overlap', 0.0),
            'continuation_cond_surprisal': proxy.get('cond_surprisal', 0.0),
            'continuation_repeat4': proxy.get('repeat4', 0.0),
            'continuation_ttr': proxy.get('ttr', 0.0),
            'continuation_composite': proxy.get('composite', 0.0),
            'continuation_composite_variance': proxy.get('composite_variance', 0.0),
            'continuation_composite_stability': proxy.get('composite_stability', 0.0),
            'continuation_improvement_rate': proxy.get('improvement_rate', 0.0),
            'continuation_ncd_matrix_mean': proxy.get('ncd_matrix_mean', 0.0),
            'continuation_ncd_matrix_variance': proxy.get('ncd_matrix_variance', 0.0),
            'continuation_ncd_matrix_min': proxy.get('ncd_matrix_min', 0.0),
        })
    else:
        result.update({
            'continuation_bscore': 0.0, 'continuation_bscore_max': 0.0,
            'continuation_determination': None, 'continuation_confidence': 0.0,
            'continuation_n_samples': 0, 'continuation_mode': None,
            'continuation_ncd': 0.0, 'continuation_internal_overlap': 0.0,
            'continuation_cond_surprisal': 0.0, 'continuation_repeat4': 0.0,
            'continuation_ttr': 0.0, 'continuation_composite': 0.0,
            'continuation_composite_variance': 0.0, 'continuation_composite_stability': 0.0,
            'continuation_improvement_rate': 0.0,
            'continuation_ncd_matrix_mean': 0.0, 'continuation_ncd_matrix_variance': 0.0,
            'continuation_ncd_matrix_min': 0.0,
        })

    # Token cohesiveness (TOCSIN)
    result.update({
        'tocsin_cohesiveness': tocsin.get('cohesiveness', 0.0),
        'tocsin_cohesiveness_std': tocsin.get('cohesiveness_std', 0.0),
        'tocsin_determination': tocsin.get('determination'),
        'tocsin_confidence': tocsin.get('confidence', 0.0),
    })

    # Perplexity compound signals (FEAT 7)
    result.update({
        'perplexity_comp_ratio': ppl.get('comp_ratio', 0.0),
        'perplexity_zlib_normalized_ppl': ppl.get('zlib_normalized_ppl', 0.0),
        'perplexity_comp_ppl_ratio': ppl.get('comp_ppl_ratio', 0.0),
    })

    # Semantic flow (inter-sentence variance)
    result.update({
        'semantic_flow_variance': semantic_flow.get('flow_variance', 0.0),
        'semantic_flow_mean': semantic_flow.get('flow_mean', 0.0),
        'semantic_flow_std': semantic_flow.get('flow_std', 0.0),
        'semantic_flow_determination': semantic_flow.get('determination'),
        'semantic_flow_confidence': semantic_flow.get('confidence', 0.0),
    })

    # Perplexity burstiness
    result.update({
        'ppl_burstiness': ppl.get('ppl_burstiness', 0.0),
        'sentence_ppl_count': ppl.get('sentence_ppl_count', 0),
    })

    # Surprisal trajectory (FEAT 10)
    result.update({
        'surprisal_trajectory_cv': surprisal_traj.get('surprisal_trajectory_cv', 0.0),
        'surprisal_var_of_var': surprisal_traj.get('surprisal_var_of_var', 0.0),
        'surprisal_stationarity': surprisal_traj.get('surprisal_stationarity', 0.0),
    })

    # Shadow model disagreement check (if memory store is active)
    shadow_disagreement = None
    if memory_store is not None:
        shadow_disagreement = memory_store.check_shadow_disagreement(result)

    result['shadow_disagreement'] = shadow_disagreement
    result['shadow_ai_prob'] = (shadow_disagreement or {}).get('shadow_ai_prob')

    return result
