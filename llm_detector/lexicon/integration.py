"""
Lexicon Packs Integration
=========================
Enhanced layer wrappers that augment the base analyzers with the externalized
versioned lexicon packs.

Usage:
    from llm_detector.lexicon.integration import (
        run_prompt_signature_enhanced,
        run_voice_dissonance_enhanced,
        run_instruction_density_enhanced,
    )
    result = run_prompt_signature_enhanced(text)
"""

import re
from typing import Dict, Optional

import llm_detector.lexicon.packs as lp

try:
    from llm_detector.analyzers.prompt_signature import run_prompt_signature
    from llm_detector.analyzers.voice_dissonance import run_voice_dissonance
    from llm_detector.analyzers.instruction_density import run_instruction_density
    from llm_detector.text_utils import get_sentences
    _HAS_ANALYZERS = True
except ImportError:
    _HAS_ANALYZERS = False


def run_prompt_signature_enhanced(text: str, base_result: Optional[dict] = None) -> dict:
    """Enhanced prompt_signature with lexicon pack integration.

    Runs the legacy prompt_signature first (or accepts pre-computed result),
    then augments with Priority 1 (constraint families) and Priority 3
    (exec-spec/rubric/Gherkin) packs.
    """
    if base_result is None:
        if not _HAS_ANALYZERS:
            raise ImportError("llm_detector.analyzers required for base prompt_signature")
        base_result = run_prompt_signature(text)

    sents = base_result.get('_sentences', None)
    if sents is None:
        if _HAS_ANALYZERS:
            sents = get_sentences(text)
        else:
            sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
            sents = [s for s in sents if s.strip()]
    n_sents = max(len(sents), 1)

    constraint_names = [n for n in lp.get_packs_for_layer('prompt_signature')
                        if lp.PACK_REGISTRY[n].category == 'constraint']
    constraint_scores = lp.score_packs(text, constraint_names, n_sents)
    total_constraint = lp.get_total_constraint_score(constraint_scores)

    exec_spec_names = [n for n in lp.get_packs_for_layer('prompt_signature')
                       if lp.PACK_REGISTRY[n].category == 'exec_spec']
    exec_spec_scores = lp.score_packs(text, exec_spec_names, n_sents)
    total_exec_spec = lp.get_total_exec_spec_score(exec_spec_scores)

    all_pack_scores = {**constraint_scores, **exec_spec_scores}
    active_families = sum(1 for s in all_pack_scores.values() if s.raw_hits > 0)

    legacy_composite = base_result.get('composite', 0.0)

    pack_boost = 0.0
    if total_constraint >= 0.40:
        pack_boost += 0.20
    elif total_constraint >= 0.20:
        pack_boost += 0.12
    elif total_constraint >= 0.08:
        pack_boost += 0.05

    if total_exec_spec >= 0.30:
        pack_boost += 0.15
    elif total_exec_spec >= 0.15:
        pack_boost += 0.08
    elif total_exec_spec >= 0.05:
        pack_boost += 0.03

    if active_families >= 6:
        pack_boost += 0.15
    elif active_families >= 4:
        pack_boost += 0.08
    elif active_families >= 2:
        pack_boost += 0.03

    uc_total = sum(s.uppercase_hits for s in all_pack_scores.values())
    if uc_total >= 3:
        pack_boost += 0.10
    elif uc_total >= 1:
        pack_boost += 0.04

    enhanced_composite = min(legacy_composite + pack_boost, 1.0)

    # Collect spans from all packs with hits
    all_pack_spans = []
    for s in all_pack_scores.values():
        if s.raw_hits > 0:
            all_pack_spans.extend(s.spans)
    all_pack_spans.sort(key=lambda x: x['start'])

    result = dict(base_result)
    result.update({
        'composite': enhanced_composite,
        'legacy_composite': legacy_composite,
        'pack_boost': round(pack_boost, 4),
        'pack_constraint_score': round(total_constraint, 4),
        'pack_exec_spec_score': round(total_exec_spec, 4),
        'pack_active_families': active_families,
        'pack_uc_hits': uc_total,
        'pack_details': {
            name: {
                'hits': s.raw_hits,
                'capped': round(s.capped_score, 4),
                'uc': s.uppercase_hits,
            }
            for name, s in all_pack_scores.items()
            if s.raw_hits > 0
        },
        'pack_spans': all_pack_spans,
    })

    return result


def run_voice_dissonance_enhanced(text: str, base_result: Optional[dict] = None) -> dict:
    """Enhanced voice_dissonance with schema/structured-output vocabulary."""
    if base_result is None:
        if not _HAS_ANALYZERS:
            raise ImportError("llm_detector.analyzers required for base voice_dissonance")
        base_result = run_voice_dissonance(text)

    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    schema_names = [n for n in lp.get_packs_for_layer('voice_dissonance')
                    if lp.PACK_REGISTRY[n].category == 'schema']
    schema_scores = lp.score_packs(text, schema_names, n_sentences=1)
    total_schema = lp.get_total_schema_score(schema_scores)

    format_names = [n for n in lp.get_packs_for_layer('voice_dissonance')
                    if lp.PACK_REGISTRY[n].category == 'format']
    format_scores = lp.score_packs(text, format_names, n_sentences=1)
    total_format = lp.get_category_score(format_scores, 'format')

    all_pack_scores = {**schema_scores, **format_scores}

    legacy_spec = base_result.get('spec_score', 0.0)
    schema_per100 = sum(s.weighted_hits for s in schema_scores.values()) / per100
    format_per100 = sum(s.weighted_hits for s in format_scores.values()) / per100
    pack_spec_boost = schema_per100 * 2.0 + format_per100 * 1.0
    enhanced_spec = legacy_spec + pack_spec_boost

    voice_score = base_result.get('voice_score', 0.0)
    enhanced_vsd = voice_score * enhanced_spec

    ssi_spec_threshold = 5.0 if base_result.get('contractions', 0) == 0 else 7.0
    enhanced_ssi = (
        enhanced_spec >= ssi_spec_threshold
        and voice_score < 0.5
        and base_result.get('hedges', 0) == 0
        and n_words >= 150
    )

    # Collect spans from all packs with hits
    all_pack_spans = []
    for s in all_pack_scores.values():
        if s.raw_hits > 0:
            all_pack_spans.extend(s.spans)
    all_pack_spans.sort(key=lambda x: x['start'])

    result = dict(base_result)
    result.update({
        'spec_score': round(enhanced_spec, 2),
        'legacy_spec_score': legacy_spec,
        'vsd': round(enhanced_vsd, 1),
        'legacy_vsd': base_result.get('vsd', 0.0),
        'ssi_enhanced': enhanced_ssi,
        'pack_schema_score': round(total_schema, 4),
        'pack_format_score': round(total_format, 4),
        'pack_schema_per100': round(schema_per100, 3),
        'pack_format_per100': round(format_per100, 3),
        'pack_spec_boost': round(pack_spec_boost, 3),
        'pack_details': {
            name: {
                'hits': s.raw_hits,
                'capped': round(s.capped_score, 4),
            }
            for name, s in all_pack_scores.items()
            if s.raw_hits > 0
        },
        'pack_spans': all_pack_spans,
    })

    return result


def run_instruction_density_enhanced(text: str, base_result: Optional[dict] = None,
                                     constraint_active: bool = False,
                                     schema_active: bool = False) -> dict:
    """Enhanced instruction_density with typed task-verb and value-domain operators."""
    if base_result is None:
        if not _HAS_ANALYZERS:
            raise ImportError("llm_detector.analyzers required for base instruction_density")
        base_result = run_instruction_density(text)

    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    idi_names = lp.get_packs_for_layer('instruction_density')
    idi_scores = lp.score_packs(text, idi_names, n_sentences=1)

    task_verb_score = idi_scores.get('task_verbs', lp.PackScore('task_verbs', 'instruction'))
    value_domain_score = idi_scores.get('value_domain', lp.PackScore('value_domain', 'instruction'))

    tv_per100 = task_verb_score.weighted_hits / per100
    vd_per100 = value_domain_score.weighted_hits / per100

    if constraint_active or schema_active:
        tv_weight = 1.0
        pairing_label = 'paired'
    else:
        tv_weight = 0.5
        pairing_label = 'unpaired'

    pack_idi_boost = (tv_per100 * tv_weight * 1.0) + (vd_per100 * 2.0)

    legacy_idi = base_result.get('idi', 0.0)
    enhanced_idi = legacy_idi + pack_idi_boost

    # Collect spans from all packs with hits
    all_pack_spans = []
    for s in idi_scores.values():
        if s.raw_hits > 0:
            all_pack_spans.extend(s.spans)
    all_pack_spans.sort(key=lambda x: x['start'])

    result = dict(base_result)
    result.update({
        'idi': round(enhanced_idi, 1),
        'legacy_idi': legacy_idi,
        'pack_idi_boost': round(pack_idi_boost, 2),
        'pack_tv_per100': round(tv_per100, 3),
        'pack_vd_per100': round(vd_per100, 3),
        'pack_tv_pairing': pairing_label,
        'pack_tv_weight': tv_weight,
        'pack_details': {
            name: {
                'hits': s.raw_hits,
                'capped': round(s.capped_score, 4),
            }
            for name, s in idi_scores.items()
            if s.raw_hits > 0
        },
        'pack_spans': all_pack_spans,
    })

    return result
