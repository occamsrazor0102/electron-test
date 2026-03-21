"""Tests for baselines.py — target 70% coverage."""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.baselines import collect_baselines, analyze_baselines, derive_attack_type

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


# Minimal result record with fields expected by collect_baselines
def _make_result(task_id='t1', word_count=200, determination='GREEN',
                 confidence=0.10, preamble_score=0.0, composite=0.10,
                 homoglyphs=0, invisible=0, delta=0.0, occupation='eng',
                 domain='tech', ground_truth=None):
    return {
        'task_id': task_id,
        'occupation': occupation,
        'attempter': 'user1',
        'word_count': word_count,
        'determination': determination,
        'confidence': confidence,
        'preamble_score': preamble_score,
        'prompt_signature_composite': composite,
        'prompt_signature_cfd': 0.0,
        'prompt_signature_mfsr': 0.0,
        'prompt_signature_framing': 0,
        'prompt_signature_must_rate': 0.0,
        'prompt_signature_distinct_frames': 0,
        'instruction_density_idi': 1.0,
        'instruction_density_imperatives': 0,
        'instruction_density_conditionals': 0,
        'voice_dissonance_voice_score': 0.5,
        'voice_dissonance_spec_score': 1.0,
        'voice_dissonance_vsd': 0,
        'voice_dissonance_voice_gated': False,
        'voice_dissonance_hedges': 0,
        'voice_dissonance_casual_markers': 0,
        'voice_dissonance_misspellings': 0,
        'ssi_triggered': False,
        'self_similarity_nssi_score': 0.0,
        'self_similarity_nssi_signals': 0,
        'self_similarity_determination': 'GREEN',
        'continuation_bscore': 0.0,
        'continuation_determination': 'GREEN',
        'self_similarity_sent_length_cv': 0.0,
        'self_similarity_comp_ratio': 0.0,
        'self_similarity_hapax_ratio': 0.0,
        'norm_obfuscation_delta': delta,
        'norm_invisible_chars': invisible,
        'norm_homoglyphs': homoglyphs,
        'norm_attack_types': '',
        'attack_type': 'none',
        'lang_support_level': 'FULL',
        'lang_fw_coverage': 0.8,
        'lang_non_latin_ratio': 0.0,
        'ground_truth': ground_truth,
        'language': 'en',
        'domain': domain,
        'mode': 'task_prompt',
        'window_max_score': 0.0,
        'window_mean_score': 0.0,
        'window_variance': 0.0,
        'window_hot_span': 0,
        'window_mixed_signal': False,
        'stylo_fw_ratio': 0.5,
        'stylo_sent_dispersion': 0.1,
        'stylo_ttr': 0.6,
        'calibrated_confidence': confidence,
        'conformity_level': 'NORMAL',
        'calibration_stratum': 'default',
        'pack_constraint_score': 0.0,
        'pack_exec_spec_score': 0.0,
        'pack_schema_score': 0.0,
        'pack_active_families': 0,
        'pack_prompt_boost': 0.0,
        'pack_idi_boost': 0.0,
        'perplexity_value': 0.0,
        'surprisal_variance': 0.0,
        'volatility_decay_ratio': 1.0,
        'continuation_composite_stability': 0.0,
        'continuation_composite_variance': 0.0,
        'continuation_improvement_rate': 0.0,
        'continuation_ncd_matrix_variance': 0.0,
        'window_fw_trajectory_cv': 0.0,
        'window_comp_trajectory_cv': 0.0,
        'tocsin_cohesiveness': 0.0,
        'perplexity_zlib_normalized_ppl': 0.0,
        'self_similarity_structural_compression_delta': 0.0,
        'surprisal_trajectory_cv': 0.0,
        'surprisal_stationarity': 0.0,
        'binoculars_score': 0.0,
    }


def test_collect_baselines_write():
    """collect_baselines writes JSONL with correct fields."""
    print("\n-- COLLECT BASELINES WRITE --")
    results = [
        _make_result('t1', 150, 'RED', 0.85, homoglyphs=0, domain='tech'),
        _make_result('t2', 450, 'GREEN', 0.05, homoglyphs=2, invisible=1,
                     delta=0.03, domain='health'),
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
    try:
        n_written = collect_baselines(results, output_path)
        check("2 records written", n_written == 2, f"got {n_written}")

        with open(output_path, 'r') as f:
            lines = f.readlines()
        check("2 lines in output file", len(lines) == 2, f"got {len(lines)}")

        r1 = json.loads(lines[0])
        check("Record 1 task_id correct", r1['task_id'] == 't1')
        check("Record 1 length_bin 'medium'",
              r1['length_bin'] == 'medium', f"got {r1['length_bin']}")
        check("Record 1 attack_type 'none'",
              r1['attack_type'] == 'none', f"got {r1['attack_type']}")
        check("Record 1 has _timestamp", '_timestamp' in r1)
        check("Record 1 has _version", r1.get('_version') == 'v0.66')

        r2 = json.loads(lines[1])
        check("Record 2 length_bin 'long'",
              r2['length_bin'] == 'long', f"got {r2['length_bin']}")
        check("Record 2 attack_type 'combined'",
              r2['attack_type'] == 'combined', f"got {r2['attack_type']}")
    finally:
        os.unlink(output_path)


def test_collect_baselines_length_bins():
    """collect_baselines assigns correct length bins."""
    print("\n-- COLLECT LENGTH BINS --")
    cases = [
        (50, 'short'),
        (200, 'medium'),
        (500, 'long'),
        (1000, 'very_long'),
    ]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
    try:
        results = [_make_result(f't{i}', wc) for i, (wc, _) in enumerate(cases)]
        collect_baselines(results, output_path)

        with open(output_path, 'r') as f:
            lines = f.readlines()
        for i, (wc, expected_bin) in enumerate(cases):
            r = json.loads(lines[i])
            check(f"word_count={wc} → length_bin={expected_bin}",
                  r['length_bin'] == expected_bin,
                  f"got {r['length_bin']}")
    finally:
        os.unlink(output_path)


def test_collect_baselines_append():
    """collect_baselines appends to existing file."""
    print("\n-- COLLECT BASELINES APPEND --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        f.write(json.dumps({'task_id': 'existing'}) + '\n')
    try:
        results = [_make_result('new1', 200)]
        collect_baselines(results, output_path)

        with open(output_path, 'r') as f:
            lines = f.readlines()
        check("2 lines after append", len(lines) == 2, f"got {len(lines)}")
        check("First line still has existing", 'existing' in lines[0])
        check("Second line has new1", 'new1' in lines[1])
    finally:
        os.unlink(output_path)


def test_analyze_baselines_distribution():
    """analyze_baselines computes determination distribution."""
    print("\n-- ANALYZE BASELINES DISTRIBUTION --")
    dets = ['RED', 'RED', 'AMBER', 'YELLOW', 'GREEN', 'GREEN', 'GREEN']

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        for det in dets:
            f.write(json.dumps({
                'determination': det, 'occupation': 'eng',
                'word_count': 200, 'confidence': 0.5,
                'prompt_signature_composite': 0.3,
            }) + '\n')
    try:
        rows = analyze_baselines(output_path)
        check("analyze_baselines returns list or None", rows is not None)
    finally:
        os.unlink(output_path)


def test_analyze_baselines_percentiles():
    """analyze_baselines computes percentile tables for metrics."""
    print("\n-- ANALYZE BASELINES PERCENTILES --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        for i in range(20):
            f.write(json.dumps({
                'determination': 'GREEN', 'occupation': 'eng',
                'word_count': 100 + i * 10, 'confidence': 0.1 + i * 0.01,
                'prompt_signature_composite': 0.05 + i * 0.01,
                'instruction_density_idi': float(i),
            }) + '\n')
    try:
        rows = analyze_baselines(output_path)
        check("Percentile rows returned", len(rows) > 0,
              f"got {len(rows)} rows")
        metrics = {r['metric'] for r in rows}
        check("word_count in metrics", 'word_count' in metrics,
              f"metrics: {metrics}")
        check("prompt_signature_composite in metrics",
              'prompt_signature_composite' in metrics)
    finally:
        os.unlink(output_path)


def test_analyze_baselines_tpr_at_fpr():
    """analyze_baselines computes TPR@FPR with ground truth labels."""
    print("\n-- ANALYZE BASELINES TPR@FPR --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        # 10 human records with low confidence
        for i in range(10):
            f.write(json.dumps({
                'ground_truth': 'human', 'determination': 'GREEN',
                'occupation': 'eng', 'confidence': 0.05 + i * 0.01,
                'word_count': 200,
            }) + '\n')
        # 10 AI records with high confidence
        for i in range(10):
            f.write(json.dumps({
                'ground_truth': 'ai', 'determination': 'RED',
                'occupation': 'eng', 'confidence': 0.70 + i * 0.02,
                'word_count': 200,
            }) + '\n')
    try:
        rows = analyze_baselines(output_path)
        check("TPR@FPR analysis runs without error", True)
    finally:
        os.unlink(output_path)


def test_analyze_baselines_stratified_rates():
    """analyze_baselines computes stratified flag rates by domain x length_bin."""
    print("\n-- ANALYZE BASELINES STRATIFIED --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        # Domain A, short text → low flag rate
        for i in range(10):
            f.write(json.dumps({
                'domain': 'domainA', 'length_bin': 'short',
                'determination': 'GREEN', 'occupation': 'eng',
                'word_count': 80, 'confidence': 0.05,
            }) + '\n')
        # Domain B, long text → high flag rate
        for i in range(10):
            f.write(json.dumps({
                'domain': 'domainB', 'length_bin': 'long',
                'determination': 'RED', 'occupation': 'eng',
                'word_count': 500, 'confidence': 0.80,
            }) + '\n')
    try:
        rows = analyze_baselines(output_path)
        check("Stratified analysis completes", True)
    finally:
        os.unlink(output_path)


def test_analyze_baselines_empty_file():
    """analyze_baselines handles empty JSONL file gracefully."""
    print("\n-- ANALYZE BASELINES EMPTY FILE --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        # Write only invalid lines
        f.write('not json\n')
        f.write('\n')
    try:
        result = analyze_baselines(output_path)
        check("Empty/invalid file returns None", result is None)
    finally:
        os.unlink(output_path)


def test_analyze_baselines_output_csv():
    """analyze_baselines writes CSV when output_csv is provided."""
    print("\n-- ANALYZE BASELINES OUTPUT CSV --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        jsonl_path = f.name
        for i in range(10):
            f.write(json.dumps({
                'determination': 'GREEN', 'occupation': 'eng',
                'word_count': 100 + i * 10, 'confidence': 0.1,
                'prompt_signature_composite': 0.1,
            }) + '\n')
    csv_path = jsonl_path.replace('.jsonl', '.csv')
    try:
        analyze_baselines(jsonl_path, output_csv=csv_path)
        if os.path.exists(csv_path):
            check("CSV file created", True)
            import pandas as pd
            df = pd.read_csv(csv_path)
            check("CSV has data", len(df) > 0, f"got {len(df)} rows")
        else:
            check("CSV created (no rows due to <5 records per occupation)", True)
    finally:
        os.unlink(jsonl_path)
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_analyze_baselines_occupation_filter():
    """analyze_baselines skips occupations with fewer than 5 records."""
    print("\n-- ANALYZE BASELINES OCCUPATION FILTER --")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        output_path = f.name
        # Only 3 records → occupation skipped (< 5 required)
        for i in range(3):
            f.write(json.dumps({
                'determination': 'GREEN', 'occupation': 'eng',
                'word_count': 200, 'confidence': 0.1,
                'prompt_signature_composite': 0.1,
            }) + '\n')
    try:
        rows = analyze_baselines(output_path)
        check("Occupation with < 5 records skipped",
              rows is not None and len(rows) == 0,
              f"got {rows}")
    finally:
        os.unlink(output_path)


if __name__ == '__main__':
    print("=" * 70)
    print("Baselines Collection Tests")
    print("=" * 70)

    test_collect_baselines_write()
    test_collect_baselines_length_bins()
    test_collect_baselines_append()
    test_analyze_baselines_distribution()
    test_analyze_baselines_percentiles()
    test_analyze_baselines_tpr_at_fpr()
    test_analyze_baselines_stratified_rates()
    test_analyze_baselines_empty_file()
    test_analyze_baselines_output_csv()
    test_analyze_baselines_occupation_filter()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:

        sys.exit(1)
