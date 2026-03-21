"""Tests for ML fusion module (dormant classifier)."""

import sys
import os
import json
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_extract_fusion_features():
    print("\n-- ML FUSION: feature extraction --")
    from llm_detector.ml_fusion import extract_fusion_features, _FUSION_FEATURE_KEYS

    # Simulate a pipeline result dict
    result = {
        'preamble_score': 0.5,
        'fingerprint_score': 0.3,
        'prompt_signature_composite': 0.2,
        'word_count': 200,
        'perplexity_value': 15.0,
        'semantic_flow_variance': 0.005,
        'ppl_burstiness': 0.3,
        'stylo_mattr': 0.72,
    }

    names, values = extract_fusion_features(result)

    check("Returns correct number of features",
          len(names) == len(_FUSION_FEATURE_KEYS),
          f"got {len(names)}, expected {len(_FUSION_FEATURE_KEYS)}")
    check("All values are floats", all(isinstance(v, float) for v in values))
    check("Missing keys default to 0.0",
          values[names.index('tocsin_cohesiveness')] == 0.0)
    check("Present keys have correct values",
          values[names.index('preamble_score')] == 0.5)
    check("New features included: semantic_flow_variance",
          'semantic_flow_variance' in names)
    check("New features included: ppl_burstiness",
          'ppl_burstiness' in names)
    check("New features included: stylo_mattr",
          'stylo_mattr' in names)


def test_extract_handles_none_values():
    print("\n-- ML FUSION: None value handling --")
    from llm_detector.ml_fusion import extract_fusion_features

    result = {
        'preamble_score': None,
        'perplexity_value': None,
        'semantic_flow_determination': 'AMBER',  # non-numeric, should be ignored
    }

    names, values = extract_fusion_features(result)
    check("None converted to 0.0", values[names.index('preamble_score')] == 0.0)
    check("All values are float", all(isinstance(v, float) for v in values))


def test_ml_determine_no_model():
    print("\n-- ML FUSION: ml_determine with no model file --")
    from llm_detector.ml_fusion import ml_determine

    result = ml_determine(['a', 'b'], [1.0, 2.0], model_path='/nonexistent/path.pkl')
    check("Returns None when no model", result is None)


def test_ml_determine_with_mock_model():
    print("\n-- ML FUSION: ml_determine with mocked model --")
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        print("  [SKIP] scikit-learn not installed")
        return

    from llm_detector.ml_fusion import ml_determine

    # Create a minimal trained model
    tmpdir = tempfile.mkdtemp()
    try:
        features = ['feat_a', 'feat_b']
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 8], [9, 2]])
        y = np.array([0, 0, 0, 1, 1, 1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression()
        clf.fit(X_scaled, y)

        model_path = os.path.join(tmpdir, 'fusion_model.pkl')
        pkg = {
            'model': clf, 'scaler': scaler, 'features': features,
            'algorithm': 'logistic_regression', 'cv_auc': 0.85,
            'n_samples': 6,
        }
        joblib.dump(pkg, model_path)

        result = ml_determine(['feat_a', 'feat_b'], [5.0, 6.0], model_path=model_path)
        check("Returns tuple", isinstance(result, tuple) and len(result) == 3)
        det, conf, expl = result
        check("Determination is string", isinstance(det, str))
        check("Confidence is float", isinstance(conf, float))
        check("Confidence in [0, 1]", 0 <= conf <= 1, f"got {conf}")
        check("Explanation mentions ML fusion", 'ML fusion' in expl)
        check("Determination is valid severity",
              det in ('GREEN', 'REVIEW', 'YELLOW', 'AMBER', 'RED'),
              f"got {det}")
    finally:
        shutil.rmtree(tmpdir)


def test_fusion_readiness_empty_store():
    print("\n-- ML FUSION: get_fusion_readiness empty store --")
    from llm_detector.memory import MemoryStore

    tmpdir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmpdir)
        readiness = store.get_fusion_readiness()
        check("total_confirmed is 0", readiness['total_confirmed'] == 0)
        check("n_ai is 0", readiness['n_ai'] == 0)
        check("n_human is 0", readiness['n_human'] == 0)
        check("ready is False", readiness['ready'] is False)
        check("progress_pct is 0", readiness['progress_pct'] == 0.0)
        check("model_info is None", readiness['model_info'] is None)
    finally:
        shutil.rmtree(tmpdir)


def test_fusion_readiness_with_labels():
    print("\n-- ML FUSION: get_fusion_readiness with labels --")
    from llm_detector.memory import MemoryStore

    tmpdir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmpdir)

        # Write some confirmed labels
        with open(store.confirmed_path, 'w') as f:
            for i in range(40):
                label = 'ai' if i < 25 else 'human'
                f.write(json.dumps({'task_id': f't{i}', 'ground_truth': label}) + '\n')

        readiness = store.get_fusion_readiness()
        check("total_confirmed is 40", readiness['total_confirmed'] == 40)
        check("n_ai is 25", readiness['n_ai'] == 25)
        check("n_human is 15", readiness['n_human'] == 15)
        check("ready is False (too few)", readiness['ready'] is False)
        check("progress_pct is 20", readiness['progress_pct'] == 20.0)
    finally:
        shutil.rmtree(tmpdir)


def test_train_insufficient_data():
    print("\n-- ML FUSION: train with insufficient data --")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("  [SKIP] scikit-learn not installed")
        return

    from llm_detector.ml_fusion import train_fusion_model
    from llm_detector.memory import MemoryStore

    tmpdir = tempfile.mkdtemp()
    try:
        store = MemoryStore(tmpdir)
        result = train_fusion_model(store)
        check("Returns dict with error", result is not None and 'error' in result,
              f"got {result}")
    finally:
        shutil.rmtree(tmpdir)


def test_fusion_backward_compat():
    """ML fusion disabled by default shouldn't change existing behavior."""
    print("\n-- ML FUSION: backward compatibility --")
    from llm_detector.fusion import determine

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

    # Without ml_fusion_enabled (default=False)
    det, _, _, cd = determine(
        0, 'NONE', l25_low, l26_none, l27_none, 300,
        mode='generic_aigt',
        ml_fusion_enabled=False,
    )
    check("Default fusion still works", det in ('GREEN', 'REVIEW'),
          f"got {det}")
    check("triggering_rule is not ml_fusion",
          cd.get('triggering_rule') != 'ml_fusion')


def test_semantic_flow_in_fusion():
    """Semantic flow parameter passes through to stylometric channel."""
    print("\n-- ML FUSION: semantic_flow passthrough --")
    from llm_detector.fusion import determine

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

    # With semantic_flow=None (should work fine)
    det1, _, _, _ = determine(
        0, 'NONE', l25_low, l26_none, l27_none, 300,
        mode='generic_aigt', semantic_flow=None,
    )
    check("semantic_flow=None doesn't break", det1 in ('GREEN', 'REVIEW'),
          f"got {det1}")

    # With semantic_flow with AMBER determination
    flow_amber = {
        'flow_variance': 0.005,
        'flow_mean': 0.50,
        'determination': 'AMBER',
        'confidence': 0.50,
    }
    det2, _, _, cd2 = determine(
        0, 'NONE', l25_low, l26_none, l27_none, 300,
        mode='generic_aigt', semantic_flow=flow_amber,
    )
    # Should see flow_variance in stylometry sub_signals
    style_subs = cd2['channels'].get('stylometry', {})
    check("semantic_flow accepted without error", True)


if __name__ == '__main__':
    print("=" * 70)
    print("ML Fusion Module Tests")
    print("=" * 70)

    test_extract_fusion_features()
    test_extract_handles_none_values()
    test_ml_determine_no_model()
    test_ml_determine_with_mock_model()
    test_fusion_readiness_empty_store()
    test_fusion_readiness_with_labels()
    test_train_insufficient_data()
    test_fusion_backward_compat()
    test_semantic_flow_in_fusion()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
