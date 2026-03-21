"""ML-based evidence fusion (dormant until sufficient training data).

Provides a trained classifier alternative to the heuristic if/else rules in
fusion.py.  Disabled by default — enable via GUI toggle once enough confirmed
labels have been collected in the MemoryStore.

Training requires >= 200 confirmed labels with at least 30 of each class.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Severity mapping for classifier output
_SEV_MAP = {0: 'GREEN', 1: 'YELLOW', 2: 'AMBER', 3: 'RED'}
_SEV_INV = {'GREEN': 0, 'YELLOW': 1, 'AMBER': 2, 'RED': 3}

# Features extracted from pipeline result dict for ML fusion.
# These are the numeric fields that the classifier is trained on.
_FUSION_FEATURE_KEYS = [
    # Channel scores
    'preamble_score', 'fingerprint_score',
    'prompt_signature_composite', 'prompt_signature_cfd',
    'prompt_signature_mfsr', 'prompt_signature_framing',
    'prompt_signature_conditional_density', 'prompt_signature_must_rate',
    # Instruction density
    'instruction_density_idi', 'instruction_density_imperatives',
    'instruction_density_conditionals', 'instruction_density_binary_specs',
    'instruction_density_flag_count',
    # Voice dissonance
    'voice_dissonance_voice_score', 'voice_dissonance_spec_score',
    'voice_dissonance_vsd', 'voice_dissonance_casual_markers',
    'voice_dissonance_hedges',
    # Self-similarity (NSSI)
    'self_similarity_nssi_score', 'self_similarity_nssi_signals',
    'self_similarity_formulaic_density', 'self_similarity_power_adj_density',
    'self_similarity_sent_length_cv', 'self_similarity_comp_ratio',
    'self_similarity_hapax_ratio',
    'self_similarity_structural_compression_delta',
    # Continuation (DNA-GPT)
    'continuation_bscore', 'continuation_bscore_max',
    'continuation_composite', 'continuation_composite_stability',
    # Semantic resonance
    'semantic_resonance_ai_mean', 'semantic_resonance_human_mean',
    'semantic_resonance_delta',
    # Perplexity
    'perplexity_value', 'surprisal_variance',
    'volatility_decay_ratio', 'binoculars_score',
    'perplexity_comp_ratio', 'perplexity_zlib_normalized_ppl',
    # Token cohesiveness (TOCSIN)
    'tocsin_cohesiveness', 'tocsin_cohesiveness_std',
    # Semantic flow
    'semantic_flow_variance', 'semantic_flow_mean', 'semantic_flow_std',
    # Perplexity burstiness
    'ppl_burstiness',
    # Stylometric features
    'stylo_fw_ratio', 'stylo_sent_dispersion', 'stylo_ttr',
    'stylo_avg_word_len', 'stylo_short_word_ratio', 'stylo_mattr',
    # Windowed scoring
    'window_max_score', 'window_mean_score', 'window_variance',
    'window_fw_trajectory_cv', 'window_comp_trajectory_cv',
    # Surprisal trajectory
    'surprisal_trajectory_cv', 'surprisal_var_of_var',
    # Meta
    'word_count',
]


def extract_fusion_features(result_dict):
    """Extract flat feature vector from a pipeline result dict.

    Args:
        result_dict: The dict returned by analyze_prompt().

    Returns:
        (feature_names: list[str], feature_values: list[float])
    """
    names = []
    values = []
    for key in _FUSION_FEATURE_KEYS:
        val = result_dict.get(key, 0)
        if val is None:
            val = 0.0
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        names.append(key)
        values.append(val)
    return names, values


def ml_determine(feature_names, feature_values, model_path=None):
    """Run trained ML fusion model.

    Args:
        feature_names: List of feature names (for validation).
        feature_values: List of float feature values.
        model_path: Path to .pkl model file. Defaults to .beet/fusion_model.pkl.

    Returns:
        (determination, confidence, explanation) tuple, or None if no model
        available or prediction fails.
    """
    if model_path is None:
        model_path = os.path.join('.beet', 'fusion_model.pkl')

    if not os.path.exists(model_path):
        return None

    try:
        import joblib
        import numpy as np

        pkg = joblib.load(model_path)
        model = pkg['model']
        scaler = pkg['scaler']
        trained_features = pkg['features']

        # Align features: reorder to match training order, fill missing with 0
        feat_map = dict(zip(feature_names, feature_values))
        aligned = [feat_map.get(f, 0.0) for f in trained_features]

        X = np.array(aligned).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict probability of AI
        proba = model.predict_proba(X_scaled)[0]
        ai_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Map probability to severity
        if ai_prob >= 0.85:
            det = 'RED'
        elif ai_prob >= 0.65:
            det = 'AMBER'
        elif ai_prob >= 0.40:
            det = 'YELLOW'
        elif ai_prob >= 0.15:
            det = 'REVIEW'
        else:
            det = 'GREEN'

        explanation = (f"ML fusion: P(AI)={ai_prob:.2f} "
                       f"(model: {pkg.get('algorithm', 'unknown')}, "
                       f"CV AUC={pkg.get('cv_auc', 0):.3f}, "
                       f"n={pkg.get('n_samples', '?')})")

        return det, round(ai_prob, 4), explanation

    except Exception as e:
        logger.warning("ML fusion prediction failed: %s", e)
        return None


def train_fusion_model(memory_store, min_samples=200, min_per_class=30,
                       algorithm='gradient_boosting'):
    """Train fusion classifier from confirmed labels in MemoryStore.

    Args:
        memory_store: MemoryStore instance.
        min_samples: Minimum total confirmed labels required.
        min_per_class: Minimum per class (ai/human) required.
        algorithm: 'gradient_boosting' or 'random_forest'.

    Returns:
        Training stats dict, or None if insufficient data.
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError:
        logger.error("ML fusion training requires scikit-learn, pandas, numpy, joblib")
        return None

    confirmed = memory_store._load_confirmed_labels()
    submissions = memory_store._load_submissions_by_task_id()

    labeled = []
    for conf in confirmed:
        tid = conf.get('task_id', '')
        if tid in submissions:
            record = submissions[tid].copy()
            record['ground_truth'] = conf['ground_truth']
            labeled.append(record)

    if len(labeled) < min_samples:
        return {'error': f'Need >= {min_samples} labeled examples, have {len(labeled)}',
                'n_labeled': len(labeled)}

    df = pd.DataFrame(labeled)
    ai_count = int((df['ground_truth'] == 'ai').sum())
    human_count = int((df['ground_truth'] == 'human').sum())

    if ai_count < min_per_class or human_count < min_per_class:
        return {'error': f'Class imbalance too severe (ai={ai_count}, human={human_count})',
                'n_ai': ai_count, 'n_human': human_count}

    # Use the predefined fusion feature keys, filtered to what's available
    feature_cols = [c for c in _FUSION_FEATURE_KEYS if c in df.columns]

    X = df[feature_cols].fillna(0).astype(float)
    y = (df['ground_truth'] == 'ai').astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight='balanced', random_state=42,
        )
    else:
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )

    clf.fit(X_scaled, y)

    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
    mean_auc = float(np.mean(cv_scores))

    from datetime import datetime
    pkg = {
        'model': clf,
        'scaler': scaler,
        'features': feature_cols,
        'algorithm': algorithm,
        'n_samples': len(df),
        'n_ai': ai_count,
        'n_human': human_count,
        'cv_auc': round(mean_auc, 4),
        'trained_at': datetime.now().isoformat(),
    }

    model_path = memory_store.store_dir / 'fusion_model.pkl'
    joblib.dump(pkg, model_path)

    # Feature importances
    if hasattr(clf, 'feature_importances_'):
        importances = sorted(zip(feature_cols, clf.feature_importances_),
                             key=lambda x: abs(x[1]), reverse=True)
        pkg['top_features'] = [(f, round(float(v), 4)) for f, v in importances[:15]]

    return {
        'n_samples': len(df),
        'n_ai': ai_count,
        'n_human': human_count,
        'cv_auc': round(mean_auc, 4),
        'algorithm': algorithm,
        'n_features': len(feature_cols),
        'model_path': str(model_path),
        'top_features': pkg.get('top_features', []),
    }
