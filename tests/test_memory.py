"""Tests for BEET Historical Memory Store."""

import sys
import os
import json
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.memory import MemoryStore

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


def _make_result(task_id, attempter, occupation, determination,
                 confidence=0.8, **kwargs):
    """Build a minimal pipeline result dict."""
    r = {
        'task_id': task_id,
        'attempter': attempter,
        'occupation': occupation,
        'determination': determination,
        'confidence': confidence,
        'reason': 'test',
        'word_count': 100,
        'pipeline_version': 'v0.66',
        'prompt_signature_composite': 0.5,
        'prompt_signature_cfd': 0.3,
        'prompt_signature_mfsr': 0.2,
        'prompt_signature_must_rate': 0.1,
        'instruction_density_idi': 5.0,
        'voice_dissonance_spec_score': 3.0,
        'voice_dissonance_voice_score': 0.5,
        'voice_dissonance_vsd': 8.0,
        'self_similarity_nssi_score': 0.3,
        'self_similarity_comp_ratio': 0.4,
        'self_similarity_hapax_ratio': 0.5,
        'self_similarity_sent_length_cv': 0.3,
        'window_max_score': 0.2,
        'window_mean_score': 0.1,
        'stylo_fw_ratio': 0.3,
        'stylo_ttr': 0.5,
        'stylo_sent_dispersion': 0.3,
        'channel_details': {'channels': {}},
    }
    r.update(kwargs)
    return r


def _make_store():
    """Create a temporary MemoryStore."""
    tmpdir = tempfile.mkdtemp()
    store_dir = os.path.join(tmpdir, '.beet')
    return MemoryStore(store_dir), tmpdir


def test_init_creates_directory():
    print("\n-- INIT: creates directory --")
    store, tmpdir = _make_store()
    try:
        check("store_dir exists", store.store_dir.exists())
        check("config.json exists", store.config_path.exists())
        check("calibration_history exists",
              (store.store_dir / 'calibration_history').exists())
        config = json.loads(store.config_path.read_text())
        check("config has version", 'version' in config)
        check("config total_submissions == 0", config['total_submissions'] == 0)
    finally:
        shutil.rmtree(tmpdir)


def test_record_batch_writes_submissions():
    print("\n-- RECORD BATCH: submissions --")
    store, tmpdir = _make_store()
    try:
        results = [
            _make_result('t1', 'alice', 'analyst', 'RED'),
            _make_result('t2', 'bob', 'analyst', 'GREEN'),
        ]
        text_map = {
            't1': 'You must include all required fields in the output.',
            't2': 'The quarterly report shows strong growth metrics.',
        }
        n = store.record_batch(results, text_map, batch_id='test_batch_1')
        check("2 submissions written", n == 2, f"got {n}")
        check("submissions.jsonl exists", store.submissions_path.exists())

        records = []
        with open(store.submissions_path) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        check("2 records in file", len(records) == 2, f"got {len(records)}")
        check("task_id preserved", records[0]['task_id'] == 't1')
        check("batch_id set", records[0]['batch_id'] == 'test_batch_1')
        check("has timestamp", 'timestamp' in records[0])
        check("has length_bin", 'length_bin' in records[0])
    finally:
        shutil.rmtree(tmpdir)


def test_record_batch_writes_fingerprints():
    print("\n-- RECORD BATCH: fingerprints --")
    store, tmpdir = _make_store()
    try:
        results = [_make_result('t1', 'alice', 'analyst', 'RED')]
        text_map = {'t1': 'Some text for fingerprinting purposes here.'}
        store.record_batch(results, text_map, batch_id='b1')

        check("fingerprints.jsonl exists", store.fingerprints_path.exists())
        with open(store.fingerprints_path) as f:
            fp = json.loads(f.readline().strip())
        check("has minhash_128", 'minhash_128' in fp)
        check("minhash length 128", len(fp['minhash_128']) == 128,
              f"got {len(fp.get('minhash_128', []))}")
        check("has structural_vec", 'structural_vec' in fp)
        check("has task_id", fp['task_id'] == 't1')
    finally:
        shutil.rmtree(tmpdir)


def test_attempter_profiles_updated():
    print("\n-- ATTEMPTER PROFILES --")
    store, tmpdir = _make_store()
    try:
        results = [
            _make_result('t1', 'alice', 'analyst', 'RED'),
            _make_result('t2', 'alice', 'analyst', 'GREEN'),
            _make_result('t3', 'alice', 'analyst', 'AMBER'),
        ]
        text_map = {
            't1': 'text one', 't2': 'text two', 't3': 'text three',
        }
        store.record_batch(results, text_map)

        profiles = store._load_attempter_profiles()
        check("alice profile exists", 'alice' in profiles)
        p = profiles['alice']
        check("total_submissions == 3", p['total_submissions'] == 3,
              f"got {p['total_submissions']}")
        check("flag_rate == 0.667", abs(p['flag_rate'] - 0.667) < 0.01,
              f"got {p['flag_rate']}")
        check("risk_tier HIGH (flag_rate > 0.30)", p['risk_tier'] == 'HIGH',
              f"got {p['risk_tier']}")
        check("has first_seen", 'first_seen' in p)
        check("has last_seen", 'last_seen' in p)
    finally:
        shutil.rmtree(tmpdir)


def test_attempter_history_query():
    print("\n-- ATTEMPTER HISTORY QUERY --")
    store, tmpdir = _make_store()
    try:
        results = [
            _make_result('t1', 'alice', 'analyst', 'RED'),
            _make_result('t2', 'alice', 'analyst', 'GREEN'),
        ]
        text_map = {'t1': 'text one', 't2': 'text two'}
        store.record_batch(results, text_map)

        history = store.get_attempter_history('alice')
        check("profile present", history['profile'] is not None)
        check("submissions count == 2", len(history['submissions']) == 2,
              f"got {len(history['submissions'])}")
        check("confirmations empty initially", len(history['confirmations']) == 0)
    finally:
        shutil.rmtree(tmpdir)


def test_risk_report_ranking():
    print("\n-- RISK REPORT RANKING --")
    store, tmpdir = _make_store()
    try:
        # Batch 1: alice = 100% flag rate, bob = 0%
        results = [
            _make_result('t1', 'alice', 'analyst', 'RED'),
            _make_result('t2', 'alice', 'analyst', 'RED'),
            _make_result('t3', 'bob', 'analyst', 'GREEN'),
            _make_result('t4', 'bob', 'analyst', 'GREEN'),
        ]
        text_map = {f't{i}': f'text {i}' for i in range(1, 5)}
        store.record_batch(results, text_map)

        report = store.get_attempter_risk_report(min_submissions=2)
        check("2 attempters in report", len(report) == 2, f"got {len(report)}")
        check("alice first (higher risk)", report[0]['attempter'] == 'alice',
              f"got {report[0]['attempter']}")
        check("alice tier HIGH", report[0]['risk_tier'] == 'HIGH',
              f"got {report[0]['risk_tier']}")
        check("bob tier NORMAL", report[1]['risk_tier'] == 'NORMAL',
              f"got {report[1]['risk_tier']}")
    finally:
        shutil.rmtree(tmpdir)


def test_record_confirmation():
    print("\n-- RECORD CONFIRMATION --")
    store, tmpdir = _make_store()
    try:
        results = [_make_result('t1', 'alice', 'analyst', 'AMBER')]
        text_map = {'t1': 'some text'}
        store.record_batch(results, text_map)

        store.record_confirmation('t1', 'ai', verified_by='reviewer_A',
                                  notes='Clear GPT pattern')

        check("confirmed.jsonl exists", store.confirmed_path.exists())
        with open(store.confirmed_path) as f:
            conf = json.loads(f.readline().strip())
        check("ground_truth == ai", conf['ground_truth'] == 'ai')
        check("verified_by set", conf['verified_by'] == 'reviewer_A')
        check("has notes", conf['notes'] == 'Clear GPT pattern')
        check("attempter captured", conf['attempter'] == 'alice')

        # Check attempter profile updated
        profiles = store._load_attempter_profiles()
        check("confirmed_ai incremented", profiles['alice']['confirmed_ai'] == 1,
              f"got {profiles['alice'].get('confirmed_ai')}")

        # Config updated
        config = json.loads(store.config_path.read_text())
        check("total_confirmed == 1", config['total_confirmed'] == 1)
    finally:
        shutil.rmtree(tmpdir)


def test_cross_batch_similarity():
    print("\n-- CROSS-BATCH SIMILARITY --")
    store, tmpdir = _make_store()
    try:
        identical = "This is the exact same text used in both batches for testing."

        # Batch 1
        results1 = [_make_result('t1', 'alice', 'analyst', 'RED')]
        store.record_batch(results1, {'t1': identical}, batch_id='batch_1')

        # Batch 2 — same text, different attempter
        results2 = [_make_result('t2', 'bob', 'analyst', 'GREEN')]
        flags = store.cross_batch_similarity(results2, {'t2': identical})

        check("cross-batch match found", len(flags) >= 1, f"got {len(flags)}")
        if flags:
            check("current_id == t2", flags[0]['current_id'] == 't2')
            check("historical_id == t1", flags[0]['historical_id'] == 't1')
            check("minhash_similarity == 1.0", flags[0]['minhash_similarity'] == 1.0,
                  f"got {flags[0]['minhash_similarity']}")
    finally:
        shutil.rmtree(tmpdir)


def test_pre_batch_context():
    print("\n-- PRE-BATCH CONTEXT --")
    store, tmpdir = _make_store()
    try:
        # Record enough data for occupation baselines
        results = [
            _make_result(f't{i}', f'worker_{i}', 'pharmacist', 'GREEN',
                         prompt_signature_cfd=0.2 + i * 0.01,
                         instruction_density_idi=3.0 + i * 0.5)
            for i in range(6)
        ]
        results[0]['determination'] = 'RED'
        results[0]['attempter'] = 'flagged_worker'
        text_map = {f't{i}': f'text {i}' for i in range(6)}
        store.record_batch(results, text_map)

        ctx = store.pre_batch_context(
            attempter='flagged_worker', occupation='pharmacist')
        check("has attempter_risk_tier", 'attempter_risk_tier' in ctx,
              f"keys: {list(ctx.keys())}")
        check("has occupation_n", 'occupation_n' in ctx,
              f"keys: {list(ctx.keys())}")
        if 'occupation_n' in ctx:
            check("occupation_n >= 5", ctx['occupation_n'] >= 5,
                  f"got {ctx['occupation_n']}")
            check("has occupation_median_cfd", 'occupation_median_cfd' in ctx)
    finally:
        shutil.rmtree(tmpdir)


def test_config_stats_updated():
    print("\n-- CONFIG STATS --")
    store, tmpdir = _make_store()
    try:
        results = [
            _make_result('t1', 'alice', 'analyst', 'RED'),
            _make_result('t2', 'bob', 'writer', 'GREEN'),
        ]
        text_map = {'t1': 'text one', 't2': 'text two'}
        store.record_batch(results, text_map)

        config = json.loads(store.config_path.read_text())
        check("total_submissions == 2", config['total_submissions'] == 2,
              f"got {config['total_submissions']}")
        check("total_batches == 1", config['total_batches'] == 1,
              f"got {config['total_batches']}")
        check("total_attempters == 2", config['total_attempters'] == 2,
              f"got {config['total_attempters']}")
        check("occupations include analyst", 'analyst' in config['occupations'])
        check("occupations include writer", 'writer' in config['occupations'])

        # Second batch
        results2 = [_make_result('t3', 'carol', 'nurse', 'YELLOW')]
        store.record_batch(results2, {'t3': 'text three'})
        config2 = json.loads(store.config_path.read_text())
        check("total_submissions == 3", config2['total_submissions'] == 3)
        check("total_batches == 2", config2['total_batches'] == 2)
    finally:
        shutil.rmtree(tmpdir)


def test_empty_store_queries():
    print("\n-- EMPTY STORE QUERIES --")
    store, tmpdir = _make_store()
    try:
        history = store.get_attempter_history('nonexistent')
        check("no profile for nonexistent", history['profile'] is None)
        check("no submissions", len(history['submissions']) == 0)

        report = store.get_attempter_risk_report()
        check("empty risk report", len(report) == 0)

        baselines = store.get_occupation_baselines('unknown_occ')
        check("empty baselines", len(baselines) == 0)

        flags = store.cross_batch_similarity([], {})
        check("empty cross-batch", len(flags) == 0)

        ctx = store.pre_batch_context(attempter='nobody', occupation='nothing')
        check("empty context", ctx == {})
    finally:
        shutil.rmtree(tmpdir)


def test_shadow_model_insufficient_data():
    print("\n-- SHADOW MODEL: insufficient data --")
    store, tmpdir = _make_store()
    try:
        # No confirmed labels at all
        result = store.rebuild_shadow_model()
        check("returns None with no data", result is None)
    finally:
        shutil.rmtree(tmpdir)


def test_shadow_disagreement_no_model():
    print("\n-- SHADOW DISAGREEMENT: no model --")
    store, tmpdir = _make_store()
    try:
        r = _make_result('t1', 'alice', 'analyst', 'RED')
        disagreement = store.check_shadow_disagreement(r)
        check("returns None when no model exists", disagreement is None)
    finally:
        shutil.rmtree(tmpdir)


def test_lexicon_discovery_no_labels():
    print("\n-- LEXICON DISCOVERY: no labels --")
    store, tmpdir = _make_store()
    try:
        # Create a dummy corpus file
        corpus_path = os.path.join(tmpdir, 'corpus.jsonl')
        with open(corpus_path, 'w') as f:
            f.write(json.dumps({'task_id': 't1', 'text': 'some text here'}) + '\n')

        candidates = store.discover_lexicon_candidates(corpus_path)
        check("returns empty with no labels", len(candidates) == 0)
    finally:
        shutil.rmtree(tmpdir)


def test_lexicon_discovery_with_data():
    print("\n-- LEXICON DISCOVERY: with labeled data --")
    store, tmpdir = _make_store()
    try:
        # Create confirmed labels
        for i in range(20):
            with open(store.confirmed_path, 'a') as f:
                label = 'ai' if i < 10 else 'human'
                f.write(json.dumps({
                    'task_id': f't{i}',
                    'ground_truth': label,
                }) + '\n')

        # Create corpus with distinct vocabulary
        corpus_path = os.path.join(tmpdir, 'corpus.jsonl')
        with open(corpus_path, 'w') as f:
            for i in range(10):
                # AI texts: heavy on "comprehensive", "furthermore", "leverage"
                ai_text = ' '.join(['comprehensive', 'furthermore', 'leverage',
                                    'optimal', 'alignment'] * 20 +
                                   ['the', 'and', 'for'] * 30)
                f.write(json.dumps({
                    'task_id': f't{i}',
                    'text': ai_text,
                }) + '\n')
            for i in range(10, 20):
                # Human texts: heavy on "honestly", "kinda", "stuff"
                human_text = ' '.join(['honestly', 'kinda', 'stuff',
                                       'yeah', 'lol'] * 20 +
                                      ['the', 'and', 'for'] * 30)
                f.write(json.dumps({
                    'task_id': f't{i}',
                    'text': human_text,
                }) + '\n')

        candidates = store.discover_lexicon_candidates(corpus_path, min_count=5)
        check("found candidates", len(candidates) > 0, f"got {len(candidates)}")
        if candidates:
            check("top candidate is AI-associated",
                  candidates[0]['log_odds'] > 0,
                  f"got log_odds={candidates[0]['log_odds']}")
            # Check CSV output was saved
            discovery_dir = store.store_dir / 'lexicon_discovery'
            csv_files = list(discovery_dir.glob('candidates_*.csv'))
            check("CSV saved", len(csv_files) == 1, f"got {len(csv_files)} files")
    finally:
        shutil.rmtree(tmpdir)


def test_shadow_model_attempter_tracking():
    print("\n-- SHADOW MODEL: attempter tracking --")
    store, tmpdir = _make_store()
    try:
        results = [
            _make_result('t1', 'alice', 'analyst', 'GREEN',
                         shadow_disagreement={
                             'type': 'model_flags_rule_passes',
                             'rule_determination': 'GREEN',
                             'shadow_ai_prob': 0.95,
                         }),
        ]
        text_map = {'t1': 'text one'}
        store.record_batch(results, text_map)

        profiles = store._load_attempter_profiles()
        check("shadow_model_flags tracked",
              profiles['alice'].get('shadow_model_flags', 0) == 1,
              f"got {profiles['alice'].get('shadow_model_flags')}")
    finally:
        shutil.rmtree(tmpdir)


def test_load_helpers():
    print("\n-- LOAD HELPERS --")
    store, tmpdir = _make_store()
    try:
        # Record some data and confirm it
        results = [
            _make_result('t1', 'alice', 'analyst', 'RED'),
            _make_result('t2', 'bob', 'analyst', 'GREEN'),
        ]
        text_map = {'t1': 'text one', 't2': 'text two'}
        store.record_batch(results, text_map)
        store.record_confirmation('t1', 'ai', verified_by='reviewer')

        confirmed = store._load_confirmed_labels()
        check("confirmed labels loaded", len(confirmed) == 1)
        check("confirmed task_id", confirmed[0]['task_id'] == 't1')

        subs = store._load_submissions_by_task_id()
        check("submissions loaded", len(subs) == 2)
        check("t1 in submissions", 't1' in subs)
        check("t2 in submissions", 't2' in subs)
    finally:
        shutil.rmtree(tmpdir)


def test_load_attempter_profiles_non_dict_json():
    """Verify _load_attempter_profiles handles non-dict JSON lines gracefully."""
    print("\n-- MEMORY: load attempter profiles with non-dict JSON --")
    tmpdir = tempfile.mkdtemp()
    try:
        store = MemoryStore(store_dir=tmpdir)
        # Write mixed JSONL: valid profile, non-dict list, bare string, number
        with open(store.attempters_path, 'w') as f:
            f.write(json.dumps({"attempter": "alice", "flag_rate": 0.5}) + "\n")
            f.write(json.dumps([1, 2, 3]) + "\n")         # list, not dict
            f.write(json.dumps("just a string") + "\n")    # string, not dict
            f.write(json.dumps(42) + "\n")                 # int, not dict
            f.write(json.dumps({"attempter": "bob", "flag_rate": 0.3}) + "\n")
        profiles = store._load_attempter_profiles()
        check("Two valid profiles loaded", len(profiles) == 2,
              f"got {len(profiles)}")
        check("alice in profiles", "alice" in profiles)
        check("bob in profiles", "bob" in profiles)
    finally:
        shutil.rmtree(tmpdir)


def test_minhash_imported_from_similarity():
    """Verify memory.py uses consolidated MinHash from similarity.py."""
    print("\n-- MEMORY: MinHash imported from similarity --")
    from llm_detector import memory
    from llm_detector import similarity
    check("_shingle_fingerprint is from similarity",
          memory._shingle_fingerprint is similarity._shingle_fingerprint)
    check("_minhash_similarity is from similarity",
          memory._minhash_similarity is similarity._minhash_similarity)


if __name__ == '__main__':
    print("=" * 70)
    print("  BEET MEMORY STORE TESTS")
    print("=" * 70)

    test_init_creates_directory()
    test_record_batch_writes_submissions()
    test_record_batch_writes_fingerprints()
    test_attempter_profiles_updated()
    test_attempter_history_query()
    test_risk_report_ranking()
    test_record_confirmation()
    test_cross_batch_similarity()
    test_pre_batch_context()
    test_config_stats_updated()
    test_empty_store_queries()
    test_shadow_model_insufficient_data()
    test_shadow_disagreement_no_model()
    test_lexicon_discovery_no_labels()
    test_lexicon_discovery_with_data()
    test_shadow_model_attempter_tracking()
    test_load_helpers()
    test_load_attempter_profiles_non_dict_json()
    test_minhash_imported_from_similarity()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
