"""Tests for cross-submission similarity analysis (FEATs 11-15)."""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.similarity import (
    _word_shingles, _jaccard, _structural_similarity,
    _shingle_fingerprint, _minhash_similarity,
    analyze_similarity, apply_similarity_adjustments,
    save_similarity_store, load_similarity_store, cross_batch_similarity,
)

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


def _make_result(task_id, attempter, occupation, determination, confidence=0.8, **kwargs):
    r = {
        'task_id': task_id,
        'attempter': attempter,
        'occupation': occupation,
        'determination': determination,
        'confidence': confidence,
        'reason': 'test',
        'prompt_signature_composite': 0.5,
        'prompt_signature_cfd': 0.3,
        'prompt_signature_mfsr': 0.2,
        'prompt_signature_must_rate': 0.1,
        'instruction_density_idi': 5.0,
        'voice_dissonance_spec_score': 3.0,
        'voice_dissonance_voice_score': 0.5,
        'self_similarity_nssi_score': 0.3,
        'self_similarity_comp_ratio': 0.4,
        'self_similarity_hapax_ratio': 0.5,
        'self_similarity_sent_length_cv': 0.3,
        'window_max_score': 0.2,
        'window_mean_score': 0.1,
        'stylo_fw_ratio': 0.3,
        'stylo_ttr': 0.5,
        'stylo_sent_dispersion': 0.3,
    }
    r.update(kwargs)
    return r


def test_shingles_basic():
    print("\n-- SHINGLES: basic --")
    s = _word_shingles("the quick brown fox jumps")
    check("returns set", isinstance(s, set))
    check("non-empty", len(s) > 0)
    check("contains 3-gram", ('the', 'quick', 'brown') in s)


def test_jaccard():
    print("\n-- JACCARD --")
    a = {(1,), (2,), (3,)}
    b = {(2,), (3,), (4,)}
    j = _jaccard(a, b)
    check("jaccard == 0.5", abs(j - 0.5) < 0.001, f"got {j}")
    check("empty sets -> 0", _jaccard(set(), set()) == 0.0)
    check("identical -> 1.0", abs(_jaccard(a, a) - 1.0) < 0.001)


def test_adaptive_baselines():
    """FEAT 11: Adaptive thresholds emit baseline records."""
    print("\n-- FEAT 11: ADAPTIVE BASELINES --")
    # Create 4 results in same occupation with varied text
    results = [
        _make_result('t1', 'alice', 'analyst', 'GREEN'),
        _make_result('t2', 'bob', 'analyst', 'GREEN'),
        _make_result('t3', 'carol', 'analyst', 'YELLOW'),
        _make_result('t4', 'dave', 'analyst', 'GREEN'),
    ]
    text_map = {
        't1': "The financial report shows quarterly revenue growth of ten percent.",
        't2': "Our analysis indicates that revenue increased by twelve percent this quarter.",
        't3': "Revenue metrics demonstrate substantial growth patterns across all segments.",
        't4': "The company reported strong earnings with significant year over year improvement.",
    }
    pairs = analyze_similarity(results, text_map, adaptive=True)
    baselines = [p for p in pairs if p.get('_type') == 'baseline']
    check("baseline record emitted", len(baselines) >= 1, f"got {len(baselines)}")
    if baselines:
        b = baselines[0]
        check("baseline has jac_median", 'jac_median' in b)
        check("baseline has adaptive_jac_threshold", 'adaptive_jac_threshold' in b)
        check("baseline has sem_median", 'sem_median' in b)
        check("baseline has n_pairs", b['n_pairs'] >= 3, f"n_pairs={b['n_pairs']}")


def test_copy_paste_detection():
    """Absolute threshold still catches copy-paste."""
    print("\n-- COPY-PASTE DETECTION --")
    results = [
        _make_result('t1', 'alice', 'writer', 'GREEN'),
        _make_result('t2', 'bob', 'writer', 'GREEN'),
    ]
    identical_text = "This is the exact same text that was copied and pasted verbatim."
    text_map = {'t1': identical_text, 't2': identical_text}
    pairs = analyze_similarity(results, text_map, jaccard_threshold=0.40, adaptive=False)
    actual = [p for p in pairs if p.get('_type') != 'baseline']
    check("copy-paste flagged", len(actual) >= 1, f"got {len(actual)}")
    if actual:
        check("flag_type includes 'text'", 'text' in actual[0]['flag_type'])
        check("jaccard ~1.0", actual[0]['jaccard'] > 0.90, f"got {actual[0]['jaccard']}")


def test_instruction_factoring():
    """FEAT 15: Instruction text subtracted from shingles."""
    print("\n-- FEAT 15: INSTRUCTION FACTORING --")
    instructions = "You must write a financial analysis report covering quarterly revenue."
    results = [
        _make_result('t1', 'alice', 'analyst', 'GREEN'),
        _make_result('t2', 'bob', 'analyst', 'GREEN'),
    ]
    # Texts that share instruction-derived shingles but differ otherwise
    text_map = {
        't1': "You must write a financial analysis report covering quarterly revenue. "
              "The company showed strong performance in Q3 with rising margins.",
        't2': "You must write a financial analysis report covering quarterly revenue. "
              "Earnings were disappointing this quarter due to supply chain issues.",
    }
    # Without factoring: higher similarity (shared instruction text)
    pairs_raw = analyze_similarity(results, text_map, adaptive=False, instruction_text=None)
    raw_actual = [p for p in pairs_raw if p.get('_type') != 'baseline']
    raw_jac = raw_actual[0]['jaccard'] if raw_actual else 0.0

    # With factoring: lower similarity (instruction shingles removed)
    pairs_factored = analyze_similarity(results, text_map, adaptive=False, instruction_text=instructions)
    factored_actual = [p for p in pairs_factored if p.get('_type') != 'baseline']

    # With instructions factored out, jaccard should be lower or pair absent
    factored_jac = factored_actual[0]['jaccard'] if factored_actual else 0.0
    check("factored jaccard <= raw jaccard", factored_jac <= raw_jac,
          f"factored={factored_jac}, raw={raw_jac}")


def test_similarity_adjustments():
    """FEAT 13: apply_similarity_adjustments upgrades determinations."""
    print("\n-- FEAT 13: SIMILARITY ADJUSTMENTS --")
    results = [
        _make_result('t1', 'alice', 'analyst', 'GREEN'),
        _make_result('t2', 'bob', 'analyst', 'YELLOW'),
        _make_result('t3', 'carol', 'analyst', 'RED'),
    ]
    sim_pairs = [
        {'id_a': 't1', 'id_b': 't2', 'flag_type': 'semantic', 'jaccard': 0.3, 'semantic': 0.95},
        {'id_a': 't2', 'id_b': 't3', 'flag_type': 'text_adaptive', 'jaccard': 0.25, 'semantic': 0.0},
    ]
    results = apply_similarity_adjustments(results, sim_pairs, {})

    # t1 (GREEN) with semantic match -> YELLOW
    check("t1 upgraded GREEN->YELLOW", results[0]['determination'] == 'YELLOW',
          f"got {results[0]['determination']}")
    check("t1 has similarity_upgrade", 'similarity_upgrade' in results[0])

    # t2 (YELLOW) with semantic match -> AMBER
    check("t2 upgraded YELLOW->AMBER", results[1]['determination'] == 'AMBER',
          f"got {results[1]['determination']}")

    # t3 (RED) should not be downgraded
    check("t3 stays RED", results[2]['determination'] == 'RED')


def test_similarity_no_downgrade():
    """FEAT 13: Never downgrade."""
    print("\n-- FEAT 13: NO DOWNGRADE --")
    results = [_make_result('t1', 'alice', 'analyst', 'RED')]
    sim_pairs = [
        {'id_a': 't1', 'id_b': 't2', 'flag_type': 'text_adaptive', 'jaccard': 0.1, 'semantic': 0.0},
    ]
    results = apply_similarity_adjustments(results, sim_pairs, {})
    check("RED not downgraded", results[0]['determination'] == 'RED')


def test_minhash_fingerprint():
    """FEAT 14: MinHash fingerprinting."""
    print("\n-- FEAT 14: MINHASH --")
    text_a = "the quick brown fox jumps over the lazy dog"
    text_b = "the quick brown fox jumps over the lazy dog"
    text_c = "a completely different sentence about cats and mice"

    sh_a = _word_shingles(text_a)
    sh_b = _word_shingles(text_b)
    sh_c = _word_shingles(text_c)

    fp_a = _shingle_fingerprint(sh_a)
    fp_b = _shingle_fingerprint(sh_b)
    fp_c = _shingle_fingerprint(sh_c)

    check("fingerprint length == 128", len(fp_a) == 128)
    check("identical texts -> minhash 1.0", _minhash_similarity(fp_a, fp_b) == 1.0)
    check("different texts -> minhash < 1.0", _minhash_similarity(fp_a, fp_c) < 1.0)
    check("empty shingles -> zeros", _shingle_fingerprint(set()) == [0] * 128)


def test_similarity_store_roundtrip():
    """FEAT 14: Save and load similarity store."""
    print("\n-- FEAT 14: STORE ROUNDTRIP --")
    results = [_make_result('t1', 'alice', 'analyst', 'GREEN', word_count=50)]
    text_map = {'t1': "Some financial analysis text for testing purposes."}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        store_path = f.name

    try:
        save_similarity_store(results, text_map, store_path)
        records = load_similarity_store(store_path)
        check("1 record stored", len(records) == 1, f"got {len(records)}")
        if records:
            check("task_id preserved", records[0]['task_id'] == 't1')
            check("minhash present", 'minhash' in records[0])
            check("minhash length 128", len(records[0]['minhash']) == 128)

        # Save again -> append
        save_similarity_store(results, text_map, store_path)
        records2 = load_similarity_store(store_path)
        check("2 records after second save", len(records2) == 2)
    finally:
        os.unlink(store_path)


def test_cross_batch_similarity():
    """FEAT 14: Cross-batch comparison."""
    print("\n-- FEAT 14: CROSS-BATCH --")
    identical_text = "This is the exact same text repeated across batches for testing."

    # Create a store with historical data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        store_path = f.name

    try:
        hist_results = [_make_result('hist_1', 'olduser', 'analyst', 'GREEN', word_count=12)]
        hist_text_map = {'hist_1': identical_text}
        save_similarity_store(hist_results, hist_text_map, store_path)

        # Current batch with same text, different attempter
        current = [_make_result('curr_1', 'newuser', 'analyst', 'GREEN', word_count=12)]
        curr_text_map = {'curr_1': identical_text}

        flags = cross_batch_similarity(current, curr_text_map, store_path, minhash_threshold=0.50)
        check("cross-batch match found", len(flags) >= 1, f"got {len(flags)}")
        if flags:
            check("current_id correct", flags[0]['current_id'] == 'curr_1')
            check("historical_id correct", flags[0]['historical_id'] == 'hist_1')
            check("minhash_similarity == 1.0", flags[0]['minhash_similarity'] == 1.0,
                  f"got {flags[0]['minhash_similarity']}")
    finally:
        os.unlink(store_path)


def test_cross_batch_empty_store():
    """FEAT 14: No store file -> no flags."""
    print("\n-- FEAT 14: EMPTY STORE --")
    flags = cross_batch_similarity([], {}, '/nonexistent/path.jsonl')
    check("empty store -> no flags", flags == [])


def test_analyze_similarity_empty():
    print("\n-- EMPTY INPUT --")
    pairs = analyze_similarity([], {})
    check("empty results -> empty pairs", pairs == [])


if __name__ == '__main__':
    print("=" * 70)
    print("  SIMILARITY ANALYSIS TESTS (FEATs 11-15)")
    print("=" * 70)

    test_shingles_basic()
    test_jaccard()
    test_adaptive_baselines()
    test_copy_paste_detection()
    test_instruction_factoring()
    test_similarity_adjustments()
    test_similarity_no_downgrade()
    test_minhash_fingerprint()
    test_similarity_store_roundtrip()
    test_cross_batch_similarity()
    test_cross_batch_empty_store()
    test_analyze_similarity_empty()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
