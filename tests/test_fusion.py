"""Tests for the fusion/determine module and channel scoring."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.fusion import determine
from llm_detector.channels.stylometric import score_stylometric

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


def test_stylometry_integration():
    print("\n-- STYLOMETRY INTEGRATION (semantic + perplexity) --")

    ch_none = score_stylometric(0, None, semantic=None, ppl=None)
    check("No signals -> GREEN", ch_none.severity == 'GREEN')

    l30_r = {'determination': 'RED', 'nssi_score': 0.8, 'nssi_signals': 7, 'confidence': 0.85}
    ch_r = score_stylometric(0, l30_r, semantic=None, ppl=None)
    check("NSSI RED still works", ch_r.severity == 'RED')

    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    ch_sem = score_stylometric(0, None, semantic=l28_amber, ppl=None)
    check("Semantic AMBER alone -> AMBER", ch_sem.severity == 'AMBER',
          f"got {ch_sem.severity}")
    check("Semantic in sub_signals", 'semantic_delta' in ch_sem.sub_signals)

    ppl_yellow = {
        'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
    }
    ch_ppl = score_stylometric(0, None, semantic=None, ppl=ppl_yellow)
    check("PPL YELLOW alone -> YELLOW", ch_ppl.severity == 'YELLOW',
          f"got {ch_ppl.severity}")
    check("Perplexity in sub_signals", 'perplexity' in ch_ppl.sub_signals)

    ch_boost = score_stylometric(0, l30_r, semantic=l28_amber, ppl=None)
    check("NSSI+Semantic boost > NSSI alone", ch_boost.score > ch_r.score,
          f"boost={ch_boost.score}, alone={ch_r.score}")


def test_determine_with_new_signals():
    print("\n-- DETERMINE WITH NEW SIGNALS --")

    l25_low = {'composite': 0.05, 'framing_completeness': 0}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3}
    l27_none = {'idi': 2.0}

    det, _, _, _ = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                             mode='generic_aigt', semantic=None, ppl=None)
    check("No new signals -> GREEN", det == 'GREEN', f"got {det}")

    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    det2, _, _, cd = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                                mode='generic_aigt', semantic=l28_amber, ppl=None)
    check("Semantic AMBER -> AMBER in generic_aigt",
          det2 in ('AMBER', 'RED'), f"got {det2}")

    check("4 channels in details", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")


def test_short_text_no_false_convergence():
    print("\n-- SHORT TEXT: no false convergence from data-insufficient channels --")

    # Single YELLOW prompt_structure signal, with word_count=60 (too short for
    # continuation, NSSI, etc.). Should stay YELLOW, not escalate to AMBER
    # via convergence with zero-score channels that couldn't run.
    prompt_sig = {'composite': 0.25, 'framing_completeness': 1, 'cfd': 0.3,
                  'mfsr': 0.2, 'must_rate': 0.1, 'distinct_frames': 2,
                  'conditional_density': 0.1, 'meta_design_hits': 0,
                  'contractions': 0, 'numbered_criteria': 0,
                  'pack_constraint_score': 0, 'pack_exec_spec_score': 0,
                  'pack_boost': 0, 'pack_active_families': 0, 'pack_spans': []}
    voice_dis = {'voice_gated': False, 'vsd': 5, 'voice_score': 0.3,
                 'spec_score': 2, 'contractions': 2, 'hedges': 1,
                 'casual_markers': 1, 'misspellings': 0, 'camel_cols': 0,
                 'calcs': 0, 'pack_schema_score': 0, 'pack_spans': []}
    instr_density = {'idi': 3.0, 'imperatives': 1, 'conditionals': 0,
                     'binary_specs': 0, 'missing_refs': 0, 'flag_count': 0,
                     'pack_idi_boost': 0, 'pack_spans': []}

    det, reason, conf, cd = determine(
        0, 'NONE', prompt_sig, voice_dis, instr_density, word_count=60,
        mode='auto',
        self_sim=None,       # Can't run: needs 200+ words
        cont_result=None,    # Can't run: needs 80+ words
        semantic=None,
        ppl=None,
        tocsin=None,
    )

    check("Short text stays YELLOW (not escalated)", det in ('YELLOW', 'GREEN', 'REVIEW'),
          f"got {det}")
    check("Short text does not reach AMBER", det != 'AMBER',
          f"got {det} — false convergence with data-insufficient channels")

    # Verify data_sufficient is tracked in channel_details
    cont_info = cd['channels'].get('continuation', {})
    check("continuation marked data_sufficient=False",
          cont_info.get('data_sufficient') is False,
          f"got {cont_info.get('data_sufficient')}")

    style_info = cd['channels'].get('stylometry', {})
    check("stylometry marked data_sufficient=False (no sub-signals)",
          style_info.get('data_sufficient') is False,
          f"got {style_info.get('data_sufficient')}")


def test_channel_ablation():
    print("\n-- CHANNEL ABLATION --")

    # A prompt that triggers prompt_structure channel at YELLOW
    prompt_sig = {'composite': 0.30, 'framing_completeness': 2, 'cfd': 0.4,
                  'mfsr': 0.3, 'must_rate': 0.2, 'distinct_frames': 3,
                  'conditional_density': 0.1, 'meta_design_hits': 0,
                  'contractions': 0, 'numbered_criteria': 0,
                  'pack_constraint_score': 0, 'pack_exec_spec_score': 0,
                  'pack_boost': 0, 'pack_active_families': 0, 'pack_spans': []}
    voice_dis = {'voice_gated': False, 'vsd': 3, 'voice_score': 0.2,
                 'spec_score': 1, 'contractions': 0, 'hedges': 0,
                 'casual_markers': 0, 'misspellings': 0, 'camel_cols': 0,
                 'calcs': 0, 'pack_schema_score': 0, 'pack_spans': []}
    instr_density = {'idi': 2.0, 'imperatives': 0, 'conditionals': 0,
                     'binary_specs': 0, 'missing_refs': 0, 'flag_count': 0,
                     'pack_idi_boost': 0, 'pack_spans': []}

    # Without ablation: should be YELLOW
    det1, _, _, _ = determine(
        0, 'NONE', prompt_sig, voice_dis, instr_density, 200,
        mode='auto',
    )
    check("Without ablation: YELLOW", det1 == 'YELLOW', f"got {det1}")

    # With prompt_structure disabled: should be GREEN
    det2, _, _, cd2 = determine(
        0, 'NONE', prompt_sig, voice_dis, instr_density, 200,
        mode='auto', disabled_channels=['prompt_structure'],
    )
    check("With prompt_structure disabled: GREEN", det2 == 'GREEN', f"got {det2}")
    check("Disabled channel marked in details",
          cd2['channels']['prompt_structure'].get('disabled') is True)

    # Disable all channels — should produce GREEN or REVIEW
    det3, _, _, cd3 = determine(0, 'NONE', prompt_sig, voice_dis, instr_density, 300,
                                 mode='task_prompt',
                                 disabled_channels=['prompt_structure', 'stylometry',
                                                    'continuation', 'windowing'])
    check("All disabled -> GREEN/REVIEW", det3 in ('GREEN', 'REVIEW'), f"got {det3}")
    check("disabled_channels listed", len(cd3.get('disabled_channels', [])) == 4)


def test_short_text_adjustment():
    print("\n-- SHORT-TEXT CHANNEL WEIGHT ADJUSTMENT --")

    l25_low = {'composite': 0.05, 'framing_completeness': 0, 'cfd': 0.01,
               'mfsr': 0, 'must_rate': 0, 'distinct_frames': 0,
               'conditional_density': 0, 'meta_design_hits': 0,
               'contractions': 0, 'numbered_criteria': 0}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3,
                'casual_markers': 3, 'misspellings': 0, 'camel_cols': 0,
                'calcs': 0}
    l27_none = {'idi': 2.0, 'imperatives': 0, 'conditionals': 0,
                'binary_specs': 0, 'missing_refs': 0, 'flag_count': 0}

    # Normal text (300 words): no short-text adjustment
    _, _, _, cd_normal = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                                    mode='generic_aigt')
    check("Normal text: no short_text_adjustment",
          cd_normal.get('short_text_adjustment') == False)

    # Short text (50 words): short_text_adjustment should activate
    _, _, _, cd_short = determine(0, 'NONE', l25_low, l26_none, l27_none, 50,
                                   mode='generic_aigt')
    check("Short text: short_text_adjustment active",
          cd_short.get('short_text_adjustment') == True or
          cd_short.get('active_channels', 4) <= 2)
    check("active_channels tracked", 'active_channels' in cd_short)


def test_attack_type_derivation():
    print("\n-- ATTACK TYPE DERIVATION --")
    from llm_detector.baselines import derive_attack_type

    # No attack
    r_none = {'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0}
    check("No attack -> 'none'", derive_attack_type(r_none) == 'none')

    # Homoglyph only
    r_homo = {'norm_homoglyphs': 5, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.03}
    check("Homoglyph -> 'homoglyph'", derive_attack_type(r_homo) == 'homoglyph')

    # Zero-width only
    r_zw = {'norm_homoglyphs': 0, 'norm_invisible_chars': 10, 'norm_obfuscation_delta': 0.05}
    check("Zero-width -> 'zero_width'", derive_attack_type(r_zw) == 'zero_width')

    # Combined
    r_combined = {'norm_homoglyphs': 3, 'norm_invisible_chars': 5, 'norm_obfuscation_delta': 0.08}
    check("Combined -> 'combined'", derive_attack_type(r_combined) == 'combined')

    # Encoding (delta only, no specific type)
    r_enc = {'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.04}
    check("Encoding -> 'encoding'", derive_attack_type(r_enc) == 'encoding')

    # Below encoding threshold
    r_low = {'norm_homoglyphs': 0, 'norm_invisible_chars': 0, 'norm_obfuscation_delta': 0.01}
    check("Low delta -> 'none'", derive_attack_type(r_low) == 'none')

    # Missing fields (graceful)
    check("Missing fields -> 'none'", derive_attack_type({}) == 'none')


def test_fusion_transparency_fields():
    print("\n-- FUSION TRANSPARENCY: fusion_counts, triggering_rule, channel roles --")

    l25_low = {'composite': 0.05, 'framing_completeness': 0, 'cfd': 0.01,
               'mfsr': 0, 'must_rate': 0, 'distinct_frames': 0,
               'conditional_density': 0, 'meta_design_hits': 0,
               'contractions': 0, 'numbered_criteria': 0,
               'pack_constraint_score': 0, 'pack_exec_spec_score': 0,
               'pack_boost': 0, 'pack_active_families': 0, 'pack_spans': []}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3,
                'casual_markers': 3, 'misspellings': 0, 'camel_cols': 0,
                'calcs': 0, 'pack_schema_score': 0, 'pack_spans': []}
    l27_none = {'idi': 2.0, 'imperatives': 0, 'conditionals': 0,
                'binary_specs': 0, 'missing_refs': 0, 'flag_count': 0,
                'pack_idi_boost': 0, 'pack_spans': []}

    _, _, _, cd = determine(
        0, 'NONE', l25_low, l26_none, l27_none, 300,
        mode='generic_aigt',
    )

    # fusion_counts should always be present
    fc = cd.get('fusion_counts', {})
    check("fusion_counts present", bool(fc), f"got {fc}")
    check("n_primary_red in fusion_counts", 'n_primary_red' in fc)
    check("n_primary_amber in fusion_counts", 'n_primary_amber' in fc)
    check("n_yellow_plus in fusion_counts", 'n_yellow_plus' in fc)
    check("n_red in fusion_counts", 'n_red' in fc)

    # triggering_rule should always be present
    rule = cd.get('triggering_rule', '')
    check("triggering_rule present", bool(rule), f"got {repr(rule)}")
    check("triggering_rule is string", isinstance(rule, str))

    # Channel role should be present for each channel
    for ch_name in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
        info = cd['channels'].get(ch_name, {})
        role = info.get('role', '')
        check(f"{ch_name} has role field", role in ('primary', 'supporting'),
              f"got {repr(role)}")

    # In generic_aigt mode all channels are primary
    check("generic_aigt: prompt_structure is primary",
          cd['channels']['prompt_structure'].get('role') == 'primary')
    check("generic_aigt: stylometry is primary",
          cd['channels']['stylometry'].get('role') == 'primary')
    check("generic_aigt: continuation is primary",
          cd['channels']['continuation'].get('role') == 'primary')

    # In task_prompt mode only task_prompt-eligible channels are primary
    _, _, _, cd_tp = determine(
        0, 'NONE', l25_low, l26_none, l27_none, 300,
        mode='task_prompt',
    )
    check("task_prompt: prompt_structure is primary",
          cd_tp['channels']['prompt_structure'].get('role') == 'primary')
    check("task_prompt: continuation is primary",
          cd_tp['channels']['continuation'].get('role') == 'primary')
    check("task_prompt: stylometry is supporting",
          cd_tp['channels']['stylometry'].get('role') == 'supporting')
    check("task_prompt: windowing is supporting",
          cd_tp['channels']['windowing'].get('role') == 'supporting')


def test_triggering_rule_no_signal():
    print("\n-- TRIGGERING RULE: no signal path --")
    l25_low = {'composite': 0.01, 'framing_completeness': 0, 'cfd': 0.0,
               'mfsr': 0, 'must_rate': 0, 'distinct_frames': 0,
               'conditional_density': 0, 'meta_design_hits': 0,
               'contractions': 0, 'numbered_criteria': 0,
               'pack_constraint_score': 0, 'pack_exec_spec_score': 0,
               'pack_boost': 0, 'pack_active_families': 0, 'pack_spans': []}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 10, 'hedges': 5,
                'casual_markers': 5, 'misspellings': 0, 'camel_cols': 0,
                'calcs': 0, 'pack_schema_score': 0, 'pack_spans': []}
    l27_none = {'idi': 0.5, 'imperatives': 0, 'conditionals': 0,
                'binary_specs': 0, 'missing_refs': 0, 'flag_count': 0,
                'pack_idi_boost': 0, 'pack_spans': []}
    det, _, _, cd = determine(
        0, 'NONE', l25_low, l26_none, l27_none, 300,
        mode='generic_aigt',
    )
    check("no-signal determination is GREEN or REVIEW", det in ('GREEN', 'REVIEW'),
          f"got {det}")
    rule = cd.get('triggering_rule', '')
    check("triggering_rule set for no-signal path", bool(rule), f"got {repr(rule)}")




if __name__ == '__main__':
    print("=" * 70)
    print("Fusion / Channel Scoring Tests")
    print("=" * 70)

    test_stylometry_integration()
    test_determine_with_new_signals()
    test_short_text_no_false_convergence()
    test_channel_ablation()
    test_short_text_adjustment()
    test_attack_type_derivation()
    test_fusion_transparency_fields()
    test_triggering_rule_no_signal()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
