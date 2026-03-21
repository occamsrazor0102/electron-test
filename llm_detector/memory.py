"""BEET Historical Memory Store.

Unified persistence layer for cross-batch detection memory.
All data lives in a single directory (default .beet/).

Usage:
    store = MemoryStore('.beet/')
    store.record_batch(results, text_map, batch_id='batch_001')
    history = store.get_attempter_history('worker_42')
    cross_matches = store.cross_batch_similarity(results, text_map)
    store.record_confirmation('task_001', 'ai', verified_by='reviewer_A')
"""

import os
import csv
import json
import logging
import statistics
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

from llm_detector.baselines import _BASELINE_FIELDS
from llm_detector.similarity import (
    _word_shingles, _STRUCT_FEATURES, _shingle_fingerprint, _minhash_similarity,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persistent memory for the BEET detection pipeline."""

    def __init__(self, store_dir='.beet'):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / 'calibration_history').mkdir(exist_ok=True)

        self.submissions_path = self.store_dir / 'submissions.jsonl'
        self.fingerprints_path = self.store_dir / 'fingerprints.jsonl'
        self.attempters_path = self.store_dir / 'attempters.jsonl'
        self.confirmed_path = self.store_dir / 'confirmed.jsonl'
        self.calibration_path = self.store_dir / 'calibration.json'
        self.config_path = self.store_dir / 'config.json'

        self._config = self._load_config()
        if not self.config_path.exists():
            self._save_config()

        # Register this store's centroid directory so get_semantic_models()
        # picks up data-derived centroids from custom --memory paths.
        from llm_detector.compat import register_centroid_path
        register_centroid_path(self.store_dir)

    # ── Config ────────────────────────────────────────────────────

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {
            'version': '0.66',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_submissions': 0,
            'total_batches': 0,
            'total_attempters': 0,
            'total_confirmed': 0,
            'occupations': [],
        }

    def _save_config(self):
        self._config['last_updated'] = datetime.now().isoformat()
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    # ── Batch Recording ──────────────────────────────────────────

    def record_batch(self, results, text_map, batch_id=None):
        """Record a full batch of pipeline results to memory.

        Updates submissions, fingerprints, attempter profiles, and config.
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"

        timestamp = datetime.now().isoformat()
        n_written = 0

        # Write submissions
        with open(self.submissions_path, 'a') as f:
            for r in results:
                record = self._extract_submission_record(r, batch_id, timestamp)
                f.write(json.dumps(record) + '\n')
                n_written += 1

        # Write fingerprints
        self._write_fingerprints(results, text_map, batch_id)

        # Update attempter profiles
        self._update_attempter_profiles(results, batch_id, timestamp)

        # Update config
        self._config['total_submissions'] += n_written
        self._config['total_batches'] += 1
        occs = set(self._config.get('occupations', []))
        for r in results:
            occ = r.get('occupation', '')
            if occ:
                occs.add(occ)
        self._config['occupations'] = sorted(occs)
        self._save_config()

        print(f"  Memory: {n_written} submissions recorded to {self.store_dir}/")
        return n_written

    def _extract_submission_record(self, r, batch_id, timestamp):
        """Extract storable fields from a pipeline result."""
        record = {k: r.get(k) for k in _BASELINE_FIELDS}
        record['batch_id'] = batch_id
        record['timestamp'] = timestamp
        record['pipeline_version'] = r.get('pipeline_version', 'unknown')

        # Similarity context
        record['similarity_partners'] = r.get('similarity_partners', 0)
        record['similarity_max_jaccard'] = r.get('similarity_max_jaccard', 0.0)
        record['similarity_max_semantic'] = r.get('similarity_max_semantic', 0.0)
        if 'similarity_upgrade' in r:
            record['similarity_upgrade'] = r['similarity_upgrade']

        # Length bin
        wc = r.get('word_count', 0)
        if wc < 100:
            record['length_bin'] = 'short'
        elif wc < 300:
            record['length_bin'] = 'medium'
        elif wc < 800:
            record['length_bin'] = 'long'
        else:
            record['length_bin'] = 'very_long'

        return record

    def _write_fingerprints(self, results, text_map, batch_id):
        """Write MinHash and optional embedding fingerprints."""
        try:
            from llm_detector.compat import HAS_SEMANTIC, get_semantic_models
        except ImportError:
            HAS_SEMANTIC = False

        # Pre-compute embeddings if available
        embeddings = {}
        if HAS_SEMANTIC:
            try:
                embedder, _, _ = get_semantic_models()
                texts = []
                tids = []
                for r in results:
                    tid = r.get('task_id', '')
                    text = text_map.get(tid, '')
                    if text:
                        texts.append(text)
                        tids.append(tid)
                if texts:
                    raw_embeds = embedder.encode(texts)
                    for tid, emb in zip(tids, raw_embeds):
                        embeddings[tid] = [round(float(v), 5) for v in emb[:64]]
            except Exception as exc:
                logger.debug("Semantic embedding for fingerprints failed: %s", exc)

        with open(self.fingerprints_path, 'a') as f:
            for r in results:
                tid = r.get('task_id', '')
                text = text_map.get(tid, '')
                if not text:
                    continue

                shingles = _word_shingles(text)
                minhash = _shingle_fingerprint(shingles)
                struct_vec = {feat: r.get(feat, 0) for feat in _STRUCT_FEATURES}

                record = {
                    'task_id': tid,
                    'attempter': r.get('attempter', ''),
                    'occupation': r.get('occupation', ''),
                    'batch_id': batch_id,
                    'determination': r.get('determination', ''),
                    'minhash_128': minhash,
                    'structural_vec': struct_vec,
                }
                if tid in embeddings:
                    record['embedding_64'] = embeddings[tid]

                f.write(json.dumps(record) + '\n')

    # ── Attempter Profiles ───────────────────────────────────────

    def _update_attempter_profiles(self, results, batch_id, timestamp):
        """Update rolling attempter profiles with new batch results."""
        profiles = self._load_attempter_profiles()

        by_att = defaultdict(list)
        for r in results:
            att = r.get('attempter', '').strip()
            if att:
                by_att[att].append(r)

        for att, submissions in by_att.items():
            if att not in profiles:
                profiles[att] = {
                    'attempter': att,
                    'total_submissions': 0,
                    'determinations': {'RED': 0, 'AMBER': 0, 'YELLOW': 0,
                                       'GREEN': 0, 'MIXED': 0, 'REVIEW': 0},
                    'confirmed_ai': 0,
                    'confirmed_human': 0,
                    'occupations': [],
                    'batches': [],
                    'first_seen': timestamp,
                    'feature_sums': {},
                    'feature_counts': 0,
                }

            p = profiles[att]
            p['total_submissions'] += len(submissions)
            p['last_seen'] = timestamp
            p['last_updated'] = timestamp

            if batch_id not in p['batches']:
                p['batches'].append(batch_id)

            for r in submissions:
                det = r.get('determination', 'GREEN')
                p['determinations'][det] = p['determinations'].get(det, 0) + 1

                occ = r.get('occupation', '')
                if occ and occ not in p['occupations']:
                    p['occupations'].append(occ)

                for feat in ['prompt_signature_cfd', 'instruction_density_idi',
                             'voice_dissonance_vsd', 'voice_dissonance_spec_score',
                             'self_similarity_nssi_score']:
                    val = r.get(feat, 0)
                    if feat not in p['feature_sums']:
                        p['feature_sums'][feat] = 0.0
                    p['feature_sums'][feat] += val
                p['feature_counts'] += 1

            # Derived fields
            total = p['total_submissions']
            flagged = (p['determinations'].get('RED', 0) +
                       p['determinations'].get('AMBER', 0) +
                       p['determinations'].get('MIXED', 0))
            p['flag_rate'] = round(flagged / max(total, 1), 3)

            if p['feature_counts'] > 0:
                p['mean_features'] = {
                    k: round(v / p['feature_counts'], 3)
                    for k, v in p['feature_sums'].items()
                }

            # Risk tier
            p['risk_tier'] = self._compute_risk_tier(p)

            # Primary detection channel
            channel_counts = Counter()
            for r in submissions:
                if r.get('determination') in ('RED', 'AMBER', 'MIXED'):
                    cd = r.get('channel_details', {}).get('channels', {})
                    for ch, info in cd.items():
                        if info.get('severity') in ('RED', 'AMBER'):
                            channel_counts[ch] += 1
            if channel_counts:
                mc = channel_counts.most_common(1)
                if mc:
                    p['primary_detection_channel'] = mc[0][0]

            # Track shadow model disagreements (sophisticated cheater pattern)
            model_flags = sum(1 for r in submissions
                              if (r.get('shadow_disagreement') or {}).get('type')
                              == 'model_flags_rule_passes')
            if model_flags > 0:
                p['shadow_model_flags'] = p.get('shadow_model_flags', 0) + model_flags

        self._save_attempter_profiles(profiles)
        self._config['total_attempters'] = len(profiles)

    @staticmethod
    def _compute_risk_tier(profile):
        """Compute risk tier from flag rate and confirmation history."""
        flag_rate = profile.get('flag_rate', 0)
        confirmed_ai = profile.get('confirmed_ai', 0)

        if confirmed_ai > 0 and flag_rate > 0.50:
            return 'CRITICAL'
        elif flag_rate > 0.30 or confirmed_ai > 0:
            return 'HIGH'
        elif flag_rate > 0.15:
            return 'ELEVATED'
        else:
            return 'NORMAL'

    def _load_attempter_profiles(self):
        """Load attempter profiles dict."""
        profiles = {}
        if self.attempters_path.exists():
            with open(self.attempters_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            p = json.loads(line)
                            profiles[p['attempter']] = p
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
        return profiles

    def _save_attempter_profiles(self, profiles):
        """Save attempter profiles (full rewrite)."""
        with open(self.attempters_path, 'w') as f:
            for p in sorted(profiles.values(),
                            key=lambda x: x.get('flag_rate', 0), reverse=True):
                f.write(json.dumps(p) + '\n')

    # ── Queries ──────────────────────────────────────────────────

    def get_attempter_history(self, attempter):
        """Get full history for a specific attempter."""
        profiles = self._load_attempter_profiles()
        profile = profiles.get(attempter.strip())

        submissions = []
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('attempter', '').strip() == attempter.strip():
                            submissions.append(rec)
                    except json.JSONDecodeError:
                        continue

        confirmations = []
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('attempter', '').strip() == attempter.strip():
                            confirmations.append(rec)
                    except json.JSONDecodeError:
                        continue

        return {
            'profile': profile,
            'submissions': submissions,
            'confirmations': confirmations,
        }

    def get_attempter_risk_report(self, min_submissions=2):
        """Get all attempters ranked by risk tier and flag rate."""
        profiles = self._load_attempter_profiles()
        tier_order = {'CRITICAL': 4, 'HIGH': 3, 'ELEVATED': 2, 'NORMAL': 1}
        return sorted(
            [p for p in profiles.values()
             if p.get('total_submissions', 0) >= min_submissions],
            key=lambda p: (-tier_order.get(p.get('risk_tier', 'NORMAL'), 0),
                           -p.get('flag_rate', 0)),
        )

    def get_occupation_baselines(self, occupation):
        """Get historical feature distributions for an occupation."""
        submissions = []
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('occupation', '') == occupation:
                            submissions.append(rec)
                    except json.JSONDecodeError:
                        continue
        return submissions

    # ── Cross-Batch Similarity ───────────────────────────────────

    def cross_batch_similarity(self, current_results, text_map,
                               minhash_threshold=0.50):
        """Compare current batch against historical fingerprints."""
        historical = []
        if self.fingerprints_path.exists():
            with open(self.fingerprints_path) as f:
                for line in f:
                    try:
                        historical.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        if not historical:
            return []

        flags = []
        for r in current_results:
            tid = r.get('task_id', '')
            text = text_map.get(tid, '')
            if not text:
                continue

            current_minhash = _shingle_fingerprint(_word_shingles(text))

            for hist in historical:
                if hist.get('task_id') == tid:
                    continue

                att_curr = r.get('attempter', '').strip().lower()
                att_hist = hist.get('attempter', '').strip().lower()
                if att_curr and att_hist and att_curr == att_hist:
                    continue

                mh_sim = _minhash_similarity(
                    current_minhash, hist.get('minhash_128', []))

                if mh_sim >= minhash_threshold:
                    flags.append({
                        'current_id': tid,
                        'historical_id': hist['task_id'],
                        'current_attempter': r.get('attempter', ''),
                        'historical_attempter': hist.get('attempter', ''),
                        'occupation': r.get('occupation', ''),
                        'minhash_similarity': round(mh_sim, 3),
                        'historical_determination': hist.get('determination', '?'),
                        'historical_batch': hist.get('batch_id', '?'),
                    })

        flags.sort(key=lambda f: f['minhash_similarity'], reverse=True)
        return flags

    # ── Confirmation Feedback ────────────────────────────────────

    def record_confirmation(self, task_id, ground_truth, verified_by='',
                            notes=''):
        """Record a human-verified ground truth label."""
        # Find original submission
        original = None
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('task_id') == task_id:
                            original = rec
                            break
                    except json.JSONDecodeError:
                        continue

        record = {
            'task_id': task_id,
            'ground_truth': ground_truth,
            'verified_by': verified_by,
            'verified_at': datetime.now().isoformat(),
            'notes': notes,
        }
        if original:
            record['attempter'] = original.get('attempter', '')
            record['occupation'] = original.get('occupation', '')
            record['pipeline_determination'] = original.get('determination', '')
            record['pipeline_confidence'] = original.get('confidence', 0)

        with open(self.confirmed_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Update attempter profile
        if original and original.get('attempter') and ground_truth in ('ai', 'human'):
            profiles = self._load_attempter_profiles()
            att = original['attempter'].strip()
            if att in profiles:
                if ground_truth == 'ai':
                    profiles[att]['confirmed_ai'] = profiles[att].get(
                        'confirmed_ai', 0) + 1
                elif ground_truth == 'human':
                    profiles[att]['confirmed_human'] = profiles[att].get(
                        'confirmed_human', 0) + 1
                profiles[att]['risk_tier'] = self._compute_risk_tier(profiles[att])
                self._save_attempter_profiles(profiles)

        if ground_truth in ('ai', 'human'):
            self._config['total_confirmed'] = self._config.get(
                'total_confirmed', 0) + 1
            self._save_config()

        print(f"  Confirmed: {task_id} = {ground_truth} (by {verified_by})")

    # ── ML Fusion Readiness ─────────────────────────────────────

    def get_fusion_readiness(self, min_required=200, min_per_class=30):
        """Return stats about confirmed label availability for ML fusion training.

        Returns dict with total_confirmed, n_ai, n_human, min_required,
        min_per_class, ready (bool), progress_pct (0-100), and model_info
        (if a trained model exists).
        """
        confirmed = self._load_confirmed_labels()
        n_ai = sum(1 for c in confirmed if c.get('ground_truth') == 'ai')
        n_human = sum(1 for c in confirmed if c.get('ground_truth') == 'human')
        total = n_ai + n_human

        ready = total >= min_required and n_ai >= min_per_class and n_human >= min_per_class
        progress_pct = min(100.0, (total / min_required) * 100) if min_required > 0 else 0.0

        result = {
            'total_confirmed': total,
            'n_ai': n_ai,
            'n_human': n_human,
            'min_required': min_required,
            'min_per_class': min_per_class,
            'ready': ready,
            'progress_pct': round(progress_pct, 1),
        }

        # Check for existing trained model
        model_path = self.store_dir / 'fusion_model.pkl'
        if model_path.exists():
            try:
                import joblib
                pkg = joblib.load(model_path)
                result['model_info'] = {
                    'trained_at': pkg.get('trained_at', 'unknown'),
                    'cv_auc': pkg.get('cv_auc', 0),
                    'n_samples': pkg.get('n_samples', 0),
                    'algorithm': pkg.get('algorithm', 'unknown'),
                }
            except Exception as exc:
                logger.debug("Failed to load fusion model info: %s", exc)
                result['model_info'] = None
        else:
            result['model_info'] = None

        return result

    # ── Calibration Integration ──────────────────────────────────

    def rebuild_calibration(self):
        """Rebuild calibration table from all confirmed human submissions."""
        from llm_detector.calibration import calibrate_from_baselines, save_calibration
        import shutil
        import tempfile

        # Collect confirmed labels
        confirmed = {}
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        confirmed[rec['task_id']] = rec['ground_truth']
                    except (json.JSONDecodeError, KeyError):
                        continue

        if not confirmed:
            print("  No confirmed labels — cannot rebuild calibration")
            return None

        # Build labeled JSONL for calibration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                         delete=False) as tmp:
            if self.submissions_path.exists():
                with open(self.submissions_path) as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            tid = rec.get('task_id', '')
                            if tid in confirmed:
                                rec['ground_truth'] = confirmed[tid]
                                tmp.write(json.dumps(rec) + '\n')
                        except json.JSONDecodeError:
                            continue
            tmp_path = tmp.name

        cal = calibrate_from_baselines(tmp_path)
        os.unlink(tmp_path)

        if cal is None:
            print("  Insufficient confirmed human data for calibration")
            return None

        # Snapshot current calibration before overwriting
        if self.calibration_path.exists():
            snapshot_name = f"cal_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
            snapshot_path = self.store_dir / 'calibration_history' / snapshot_name
            shutil.copy2(self.calibration_path, snapshot_path)

        save_calibration(cal, str(self.calibration_path))
        return cal

    # ── Pipeline Integration Hook ────────────────────────────────

    def pre_batch_context(self, attempter=None, occupation=None):
        """Retrieve historical context before running a batch."""
        context = {}

        if attempter:
            profiles = self._load_attempter_profiles()
            profile = profiles.get(attempter.strip())
            if profile:
                context['attempter_risk_tier'] = profile.get('risk_tier', 'UNKNOWN')
                context['attempter_flag_rate'] = profile.get('flag_rate', 0)
                context['attempter_total'] = profile.get('total_submissions', 0)
                context['attempter_confirmed_ai'] = profile.get('confirmed_ai', 0)

        if occupation:
            subs = self.get_occupation_baselines(occupation)
            if len(subs) >= 5:
                cfd_values = [s.get('prompt_signature_cfd', 0) for s in subs]
                idi_values = [s.get('instruction_density_idi', 0) for s in subs]
                context['occupation_n'] = len(subs)
                context['occupation_median_cfd'] = round(
                    statistics.median(cfd_values), 3)
                context['occupation_median_idi'] = round(
                    statistics.median(idi_values), 3)

        return context

    # ── Summary ──────────────────────────────────────────────────

    def print_summary(self):
        """Print memory store summary."""
        c = self._config
        print(f"\n  BEET Memory Store: {self.store_dir}/")
        print(f"    Submissions: {c.get('total_submissions', 0)}")
        print(f"    Batches:     {c.get('total_batches', 0)}")
        print(f"    Attempters:  {c.get('total_attempters', 0)}")
        print(f"    Confirmed:   {c.get('total_confirmed', 0)}")
        print(f"    Occupations: {', '.join(c.get('occupations', []))}")
        print(f"    Last update: {c.get('last_updated', 'never')}")

    # ── Shadow Model (Tool 1) ────────────────────────────────────

    def _load_confirmed_labels(self):
        """Load all confirmed labels from confirmed.jsonl."""
        confirmed = []
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        confirmed.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return confirmed

    def _load_submissions_by_task_id(self):
        """Load submissions indexed by task_id."""
        submissions = {}
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        tid = rec.get('task_id', '')
                        if tid:
                            submissions[tid] = rec
                    except json.JSONDecodeError:
                        continue
        return submissions

    def rebuild_shadow_model(self):
        """Train shadow classifier from confirmed labels in memory.

        Requires >= 200 confirmed labels with reasonable class balance
        (at least 30 of each class). Uses L1-penalized logistic regression
        for interpretability.

        Saves model to .beet/shadow_model.pkl
        Returns trained model package or None if insufficient data.
        """
        try:
            import numpy as np
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            import joblib
        except ImportError:
            print("  Shadow model requires scikit-learn, pandas, numpy, joblib")
            return None

        confirmed = self._load_confirmed_labels()
        submissions = self._load_submissions_by_task_id()

        labeled = []
        for conf in confirmed:
            tid = conf.get('task_id', '')
            if tid in submissions:
                record = submissions[tid].copy()
                record['ground_truth'] = conf['ground_truth']
                labeled.append(record)

        df = pd.DataFrame(labeled)
        df = df[df['ground_truth'].isin(['ai', 'human'])]
        if len(df) < 200:
            print(f"  Shadow model: need >= 200 labeled examples, have {len(df)}")
            return None
        ai_count = (df['ground_truth'] == 'ai').sum()
        human_count = (df['ground_truth'] == 'human').sum()

        if ai_count < 30 or human_count < 30:
            print(f"  Shadow model: class imbalance too severe "
                  f"(ai={ai_count}, human={human_count}, need >= 30 each)")
            return None

        feature_cols = [c for c in df.columns
                        if df[c].dtype in ('float64', 'int64', 'float32')
                        and c not in ('word_count_raw', 'ground_truth')
                        and not c.startswith('_')]

        X = df[feature_cols].fillna(0)
        y = (df['ground_truth'] == 'ai').astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(
            penalty='l1', solver='liblinear',
            class_weight='balanced', C=1.0, max_iter=1000,
        )
        clf.fit(X_scaled, y)

        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)

        pkg = {
            'model': clf,
            'scaler': scaler,
            'features': feature_cols,
            'n_samples': len(df),
            'n_ai': int(ai_count),
            'n_human': int(human_count),
            'cv_auc': round(float(mean_auc), 4),
            'trained_at': datetime.now().isoformat(),
        }

        model_path = self.store_dir / 'shadow_model.pkl'
        joblib.dump(pkg, model_path)

        print(f"\n  Shadow model trained: {len(df)} samples, "
              f"CV AUC={mean_auc:.3f}")
        print(f"  {'Feature':<40} {'Coefficient':>12}")
        print(f"  {'-'*52}")
        coefs = sorted(zip(feature_cols, clf.coef_[0]),
                       key=lambda x: -abs(x[1]))
        for feat, coef in coefs:
            if abs(coef) > 0.01:
                direction = "-> AI" if coef > 0 else "-> Human"
                print(f"  {feat:<40} {coef:>+8.4f}  {direction}")

        n_active = sum(1 for _, c in coefs if abs(c) > 0.01)
        n_zeroed = len(feature_cols) - n_active
        print(f"\n  Active features: {n_active}/{len(feature_cols)} "
              f"({n_zeroed} zeroed by L1 penalty)")

        return pkg

    def check_shadow_disagreement(self, result):
        """Run shadow model on a pipeline result, return disagreement info.

        Returns None if no disagreement, or a dict describing the discrepancy.
        Only runs if a trained shadow model exists in the memory store.
        """
        model_path = self.store_dir / 'shadow_model.pkl'
        if not model_path.exists():
            return None

        try:
            import joblib
            import numpy as np
        except ImportError:
            return None

        if not hasattr(self, '_shadow_pkg'):
            self._shadow_pkg = joblib.load(model_path)

        pkg = self._shadow_pkg
        feature_vec = [result.get(f, 0) for f in pkg['features']]
        X = np.array([feature_vec])
        X_scaled = pkg['scaler'].transform(X)
        ai_prob = float(pkg['model'].predict_proba(X_scaled)[0][1])

        rule_det = result.get('determination', 'GREEN')
        rule_flagged = rule_det in ('RED', 'AMBER', 'MIXED')
        model_flagged = ai_prob > 0.80

        disagreement = None

        if rule_flagged and not model_flagged:
            disagreement = {
                'type': 'rule_flags_model_passes',
                'rule_determination': rule_det,
                'shadow_ai_prob': round(ai_prob, 4),
                'interpretation': 'Possible false positive -- rule engine fires '
                                'but learned model sees human patterns',
            }
        elif not rule_flagged and model_flagged:
            disagreement = {
                'type': 'model_flags_rule_passes',
                'rule_determination': rule_det,
                'shadow_ai_prob': round(ai_prob, 4),
                'interpretation': 'Possible blind spot -- learned model detects '
                                'AI patterns that rule engine misses',
            }

        if disagreement:
            log_record = {
                'task_id': result.get('task_id', ''),
                'attempter': result.get('attempter', ''),
                'timestamp': datetime.now().isoformat(),
                **disagreement,
            }
            log_path = self.store_dir / 'shadow_disagreements.jsonl'
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_record) + '\n')

        return disagreement

    # ── Lexicon Discovery (Tool 2) ────────────────────────────────

    def discover_lexicon_candidates(self, corpus_path, min_count=10,
                                     log_odds_threshold=1.5):
        """Discover AI-associated vocabulary via Smoothed Log-Odds.

        Uses confirmed ground truth labels from memory joined with raw text
        from the provided corpus file.

        Args:
            corpus_path: Path to JSONL with {"task_id": "...", "text": "..."}
            min_count: Minimum total occurrences across both classes
            log_odds_threshold: Minimum log-odds ratio to report (1.5 ~ 4.5x overuse)

        Saves candidates CSV to .beet/lexicon_discovery/
        Returns list of candidate dicts.
        """
        import math
        import csv

        confirmed = {}
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        confirmed[rec['task_id']] = rec['ground_truth']
                    except (json.JSONDecodeError, KeyError):
                        continue

        if not confirmed:
            print("  Lexicon discovery: no confirmed labels in memory")
            return []

        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    tid = rec.get('task_id', '')
                    text = rec.get('text', rec.get('prompt', ''))
                    if tid and text:
                        corpus[tid] = text.lower()
                except json.JSONDecodeError:
                    continue

        ai_words = []
        human_words = []
        for tid, label in confirmed.items():
            text = corpus.get(tid, '')
            if not text:
                continue
            words = [w for w in text.split() if w.isalpha() and len(w) > 2]
            if label == 'ai':
                ai_words.extend(words)
            elif label == 'human':
                human_words.extend(words)

        if len(ai_words) < 500 or len(human_words) < 500:
            print(f"  Lexicon discovery: insufficient text "
                  f"(ai={len(ai_words)} words, human={len(human_words)} words)")
            return []

        ai_counts = Counter(ai_words)
        human_counts = Counter(human_words)
        vocab = set(ai_counts.keys()) | set(human_counts.keys())
        n_ai = sum(ai_counts.values())
        n_human = sum(human_counts.values())

        alpha = 0.05
        candidates = []

        existing_fps = self._get_existing_fingerprints()
        existing_packs = self._get_existing_pack_keywords()

        for word in vocab:
            total_count = ai_counts[word] + human_counts[word]
            if total_count < min_count:
                continue

            ai_rate = (ai_counts[word] + alpha) / (n_ai + alpha * len(vocab))
            human_rate = (human_counts[word] + alpha) / (n_human + alpha * len(vocab))

            log_odds = math.log(ai_rate / human_rate)

            if log_odds > log_odds_threshold:
                candidates.append({
                    'word': word,
                    'log_odds': round(log_odds, 3),
                    'ai_freq': ai_counts[word],
                    'human_freq': human_counts[word],
                    'total_freq': total_count,
                    'ai_rate_per_1k': round(ai_counts[word] / (n_ai / 1000), 2),
                    'human_rate_per_1k': round(human_counts[word] / (n_human / 1000), 2),
                    'already_in_fingerprints': word in existing_fps,
                    'already_in_packs': word in existing_packs,
                })

        candidates.sort(key=lambda c: -c['log_odds'])

        discovery_dir = self.store_dir / 'lexicon_discovery'
        discovery_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d')
        out_path = discovery_dir / f'candidates_{timestamp}.csv'

        if candidates:
            with open(out_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
                writer.writeheader()
                writer.writerows(candidates)

        n_new = sum(1 for c in candidates
                    if not c['already_in_fingerprints']
                    and not c['already_in_packs'])

        print(f"\n  Lexicon discovery: {len(candidates)} candidates "
              f"({n_new} new, not in existing vocabulary)")
        if candidates:
            print(f"  Saved to {out_path}")
            print(f"\n  Top 15 candidates:")
            print(f"  {'Word':<25} {'Log-Odds':>9} {'AI/1k':>7} {'Hum/1k':>7} {'New?':>5}")
            print(f"  {'-'*55}")
            for c in candidates[:15]:
                new = '*' if not c['already_in_fingerprints'] and not c['already_in_packs'] else ''
                print(f"  {c['word']:<25} {c['log_odds']:>+8.3f} "
                      f"{c['ai_rate_per_1k']:>7.1f} {c['human_rate_per_1k']:>7.1f} "
                      f"{new:>5}")

        return candidates

    def _get_existing_fingerprints(self):
        """Get current fingerprint words for duplicate detection."""
        try:
            from llm_detector.analyzers.fingerprint import FINGERPRINT_WORDS
            return set(w.lower() for w in FINGERPRINT_WORDS)
        except (ImportError, AttributeError):
            return set()

    def _get_existing_pack_keywords(self):
        """Get all keywords from registered lexicon packs."""
        try:
            from llm_detector.lexicon.packs import PACK_REGISTRY
            all_kw = set()
            for pack in PACK_REGISTRY.values():
                all_kw.update(k.lower() for k in pack.keywords)
            return all_kw
        except (ImportError, AttributeError):
            return set()

    # ── Semantic Centroid Rebuilder (Tool 3) ───────────────────────

    def rebuild_semantic_centroids(self, corpus_path, min_per_class=50):
        """Rebuild semantic centroids from confirmed labeled text.

        Args:
            corpus_path: Path to JSONL with {"task_id": "...", "text": "..."}
            min_per_class: Minimum confirmed examples per class for safe rebuild.

        Saves versioned centroids to .beet/centroids/centroids_vXX.npz
        Returns dict with centroid arrays or None.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            print("  Centroid rebuild requires sentence-transformers and numpy")
            return None

        confirmed = {}
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        confirmed[rec['task_id']] = rec['ground_truth']
                    except (json.JSONDecodeError, KeyError):
                        continue

        ai_texts = []
        human_texts = []
        with open(corpus_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    tid = rec.get('task_id', '')
                    text = rec.get('text', rec.get('prompt', ''))
                    if tid in confirmed and text:
                        if confirmed[tid] == 'ai':
                            ai_texts.append(text)
                        elif confirmed[tid] == 'human':
                            human_texts.append(text)
                except json.JSONDecodeError:
                    continue

        if len(ai_texts) < min_per_class or len(human_texts) < min_per_class:
            print(f"  Centroid rebuild: insufficient confirmed text "
                  f"(ai={len(ai_texts)}, human={len(human_texts)}, "
                  f"need >= {min_per_class} each)")
            return None

        print(f"  Computing centroids: {len(ai_texts)} AI, "
              f"{len(human_texts)} human texts...")

        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        ai_embeddings = embedder.encode(ai_texts, show_progress_bar=True)
        human_embeddings = embedder.encode(human_texts, show_progress_bar=True)

        ai_centroid = np.mean(ai_embeddings, axis=0, keepdims=True)
        human_centroid = np.mean(human_embeddings, axis=0, keepdims=True)

        n_clusters = min(5, len(ai_texts) // 20, len(human_texts) // 20)
        n_clusters = max(1, n_clusters)

        if n_clusters > 1:
            from sklearn.cluster import KMeans
            ai_km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            ai_km.fit(ai_embeddings)
            ai_multi_centroids = ai_km.cluster_centers_

            human_km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            human_km.fit(human_embeddings)
            human_multi_centroids = human_km.cluster_centers_
        else:
            ai_multi_centroids = ai_centroid
            human_multi_centroids = human_centroid

        centroids_dir = self.store_dir / 'centroids'
        centroids_dir.mkdir(exist_ok=True)
        version = datetime.now().strftime('%Y%m%d')
        out_path = centroids_dir / f'centroids_v{version}.npz'

        np.savez(
            out_path,
            ai_centroid=ai_centroid,
            human_centroid=human_centroid,
            ai_multi=ai_multi_centroids,
            human_multi=human_multi_centroids,
            n_ai=len(ai_texts),
            n_human=len(human_texts),
            n_clusters=n_clusters,
        )

        import shutil
        latest_path = centroids_dir / 'centroids_latest.npz'
        shutil.copy2(out_path, latest_path)

        print(f"  Centroids saved: {out_path}")
        print(f"    AI: {len(ai_texts)} texts -> {n_clusters} centroid(s)")
        print(f"    Human: {len(human_texts)} texts -> {n_clusters} centroid(s)")

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        mean_sep = 1.0 - float(cos_sim(ai_centroid, human_centroid)[0][0])
        print(f"    Centroid separation: {mean_sep:.4f} "
              f"(higher = better discrimination)")

        return {
            'ai_centroids': ai_multi_centroids,
            'human_centroids': human_multi_centroids,
            'separation': mean_sep,
            'path': str(out_path),
        }

    # ── Summary ──────────────────────────────────────────────────

    def print_attempter_history(self, attempter):
        """Print formatted attempter history."""
        history = self.get_attempter_history(attempter)
        profile = history['profile']

        if not profile:
            print(f"\n  No history found for attempter: {attempter}")
            return

        p = profile
        print(f"\n{'='*70}")
        print(f"  ATTEMPTER HISTORY: {p['attempter']}")
        print(f"{'='*70}")
        print(f"    Risk tier:    {p.get('risk_tier', '?')}")
        print(f"    Submissions:  {p.get('total_submissions', 0)}")
        print(f"    Flag rate:    {p.get('flag_rate', 0):.1%}")
        print(f"    First seen:   {p.get('first_seen', '?')[:10]}")
        print(f"    Last seen:    {p.get('last_seen', '?')[:10]}")
        print(f"    Batches:      {len(p.get('batches', []))}")
        print(f"    Occupations:  {', '.join(p.get('occupations', []))}")

        dets = p.get('determinations', {})
        print(f"    Determinations: R={dets.get('RED', 0)} A={dets.get('AMBER', 0)} "
              f"Y={dets.get('YELLOW', 0)} G={dets.get('GREEN', 0)}")

        confirmed_ai = p.get('confirmed_ai', 0)
        confirmed_human = p.get('confirmed_human', 0)
        if confirmed_ai or confirmed_human:
            print(f"    Confirmed:    AI={confirmed_ai}  Human={confirmed_human}")

        if p.get('primary_detection_channel'):
            print(f"    Primary channel: {p['primary_detection_channel']}")

        subs = history['submissions']
        if subs:
            print(f"\n    Recent submissions ({len(subs)} total):")
            for s in subs[-5:]:
                print(f"      {s.get('task_id', '?')[:15]:15s} "
                      f"[{s.get('determination', '?')}] "
                      f"conf={s.get('confidence', 0):.2f} "
                      f"{s.get('occupation', '')[:25]}")

    # ── Shadow Model (Tool 1) ────────────────────────────────────

    def _load_confirmed_labels(self):
        """Load all confirmed labels."""
        confirmed = []
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        confirmed.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return confirmed

    def _load_submissions_by_task_id(self):
        """Load submissions indexed by task_id."""
        submissions = {}
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        tid = rec.get('task_id', '')
                        if tid:
                            submissions[tid] = rec
                    except json.JSONDecodeError:
                        continue
        return submissions

    def rebuild_shadow_model(self):
        """Train shadow classifier from confirmed labels in memory.

        Requires >= 200 confirmed labels with reasonable class balance
        (at least 30 of each class). Uses L1-penalized logistic regression
        for interpretability.

        Saves model to .beet/shadow_model.pkl
        Returns trained model package or None if insufficient data.
        """
        confirmed = self._load_confirmed_labels()
        submissions = self._load_submissions_by_task_id()

        # Deduplicate: keep only the latest confirmation per task_id
        latest_by_tid = {}
        for conf in confirmed:
            latest_by_tid[conf['task_id']] = conf

        labeled = []
        for tid, conf in latest_by_tid.items():
            if tid in submissions and 'ground_truth' in conf:
                record = submissions[tid].copy()
                record['ground_truth'] = conf['ground_truth']
                labeled.append(record)

        if not labeled:
            print("  Shadow model: no confirmed labels found")
            return None

        try:
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            import joblib
            import numpy as np
        except ImportError:
            print("  Shadow model requires pandas, scikit-learn, joblib, numpy")
            return None

        df = pd.DataFrame(labeled)
        df = df[df['ground_truth'].isin(['ai', 'human'])]
        if len(df) < 200:
            print(f"  Shadow model: need >= 200 labeled examples, have {len(df)}")
            return None
        ai_count = (df['ground_truth'] == 'ai').sum()
        human_count = (df['ground_truth'] == 'human').sum()

        if ai_count < 30 or human_count < 30:
            print(f"  Shadow model: class imbalance too severe "
                  f"(ai={ai_count}, human={human_count}, need >= 30 each)")
            return None

        feature_cols = [c for c in df.columns
                        if df[c].dtype in ('float64', 'int64', 'float32')
                        and c not in ('word_count_raw', 'ground_truth')
                        and not c.startswith('_')]

        X = df[feature_cols].fillna(0)
        y = (df['ground_truth'] == 'ai').astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(
            penalty='l1', solver='liblinear',
            class_weight='balanced', C=1.0, max_iter=1000,
        )
        clf.fit(X_scaled, y)

        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)

        pkg = {
            'model': clf,
            'scaler': scaler,
            'features': feature_cols,
            'n_samples': len(df),
            'n_ai': int(ai_count),
            'n_human': int(human_count),
            'cv_auc': round(float(mean_auc), 4),
            'trained_at': datetime.now().isoformat(),
        }

        model_path = self.store_dir / 'shadow_model.pkl'
        joblib.dump(pkg, model_path)

        print(f"\n  Shadow model trained: {len(df)} samples, "
              f"CV AUC={mean_auc:.3f}")
        print(f"  {'Feature':<40} {'Coefficient':>12}")
        print(f"  {'-'*52}")
        coefs = sorted(zip(feature_cols, clf.coef_[0]),
                       key=lambda x: -abs(x[1]))
        for feat, coef in coefs:
            if abs(coef) > 0.01:
                direction = "-> AI" if coef > 0 else "-> Human"
                print(f"  {feat:<40} {coef:>+8.4f}  {direction}")

        n_active = sum(1 for _, c in coefs if abs(c) > 0.01)
        n_zeroed = len(feature_cols) - n_active
        print(f"\n  Active features: {n_active}/{len(feature_cols)} "
              f"({n_zeroed} zeroed by L1 penalty)")

        return pkg

    def check_shadow_disagreement(self, result):
        """Run shadow model on a pipeline result, return disagreement info.

        Returns None if no disagreement, or a dict describing the discrepancy.
        Only runs if a trained shadow model exists in the memory store.
        """
        model_path = self.store_dir / 'shadow_model.pkl'
        if not model_path.exists():
            return None

        try:
            import joblib
            import numpy as np
        except ImportError:
            return None

        if not hasattr(self, '_shadow_pkg'):
            self._shadow_pkg = joblib.load(model_path)

        pkg = self._shadow_pkg
        feature_vec = [result.get(f, 0) for f in pkg['features']]
        X = np.array([feature_vec])
        X_scaled = pkg['scaler'].transform(X)
        ai_prob = float(pkg['model'].predict_proba(X_scaled)[0][1])

        rule_det = result.get('determination', 'GREEN')
        rule_flagged = rule_det in ('RED', 'AMBER', 'MIXED')
        model_flagged = ai_prob > 0.80

        disagreement = None

        if rule_flagged and not model_flagged:
            disagreement = {
                'type': 'rule_flags_model_passes',
                'rule_determination': rule_det,
                'shadow_ai_prob': round(ai_prob, 4),
                'interpretation': 'Possible false positive — rule engine fires '
                                'but learned model sees human patterns',
            }
        elif not rule_flagged and model_flagged:
            disagreement = {
                'type': 'model_flags_rule_passes',
                'rule_determination': rule_det,
                'shadow_ai_prob': round(ai_prob, 4),
                'interpretation': 'Possible blind spot — learned model detects '
                                'AI patterns that rule engine misses',
            }

        if disagreement:
            log_record = {
                'task_id': result.get('task_id', ''),
                'attempter': result.get('attempter', ''),
                'timestamp': datetime.now().isoformat(),
                **disagreement,
            }
            log_path = self.store_dir / 'shadow_disagreements.jsonl'
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_record) + '\n')

        return disagreement

    # ── Lexicon Discovery (Tool 2) ───────────────────────────────

    def discover_lexicon_candidates(self, corpus_path, min_count=10,
                                     log_odds_threshold=1.5):
        """Discover AI-associated vocabulary via Smoothed Log-Odds.

        Uses confirmed ground truth labels from memory joined with raw text
        from the provided corpus file.

        Args:
            corpus_path: Path to JSONL with {"task_id": "...", "text": "..."}
            min_count: Minimum total occurrences across both classes
            log_odds_threshold: Minimum log-odds ratio to report

        Saves candidates CSV to .beet/lexicon_discovery/
        Returns list of candidate dicts.
        """
        from collections import Counter
        import math
        import csv

        confirmed = {}
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        confirmed[rec['task_id']] = rec['ground_truth']
                    except (json.JSONDecodeError, KeyError):
                        continue

        if not confirmed:
            print("  Lexicon discovery: no confirmed labels in memory")
            return []

        corpus = {}
        with open(corpus_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    tid = rec.get('task_id', '')
                    text = rec.get('text', rec.get('prompt', ''))
                    if tid and text:
                        corpus[tid] = text.lower()
                except json.JSONDecodeError:
                    continue

        ai_words = []
        human_words = []
        for tid, label in confirmed.items():
            text = corpus.get(tid, '')
            if not text:
                continue
            words = [w for w in text.split() if w.isalpha() and len(w) > 2]
            if label == 'ai':
                ai_words.extend(words)
            elif label == 'human':
                human_words.extend(words)

        if len(ai_words) < 500 or len(human_words) < 500:
            print(f"  Lexicon discovery: insufficient text "
                  f"(ai={len(ai_words)} words, human={len(human_words)} words)")
            return []

        ai_counts = Counter(ai_words)
        human_counts = Counter(human_words)
        vocab = set(ai_counts.keys()) | set(human_counts.keys())
        n_ai = sum(ai_counts.values())
        n_human = sum(human_counts.values())

        alpha = 0.05
        candidates = []

        for word in vocab:
            total_count = ai_counts[word] + human_counts[word]
            if total_count < min_count:
                continue

            ai_rate = (ai_counts[word] + alpha) / (n_ai + alpha * len(vocab))
            human_rate = (human_counts[word] + alpha) / (n_human + alpha * len(vocab))

            log_odds = math.log(ai_rate / human_rate)

            if log_odds > log_odds_threshold:
                candidates.append({
                    'word': word,
                    'log_odds': round(log_odds, 3),
                    'ai_freq': ai_counts[word],
                    'human_freq': human_counts[word],
                    'total_freq': total_count,
                    'ai_rate_per_1k': round(ai_counts[word] / (n_ai / 1000), 2),
                    'human_rate_per_1k': round(human_counts[word] / (n_human / 1000), 2),
                    'already_in_fingerprints': word in self._get_existing_fingerprints(),
                    'already_in_packs': word in self._get_existing_pack_keywords(),
                })

        candidates.sort(key=lambda c: -c['log_odds'])

        discovery_dir = self.store_dir / 'lexicon_discovery'
        discovery_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d')
        out_path = discovery_dir / f'candidates_{timestamp}.csv'

        if candidates:
            with open(out_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
                writer.writeheader()
                writer.writerows(candidates)

        n_new = sum(1 for c in candidates
                    if not c['already_in_fingerprints']
                    and not c['already_in_packs'])

        print(f"\n  Lexicon discovery: {len(candidates)} candidates "
              f"({n_new} new, not in existing vocabulary)")
        if candidates:
            print(f"  Saved to {out_path}")
            print(f"\n  Top 15 candidates:")
            print(f"  {'Word':<25} {'Log-Odds':>9} {'AI/1k':>7} {'Hum/1k':>7} {'New?':>5}")
            print(f"  {'-'*55}")
            for c in candidates[:15]:
                new = '*' if not c['already_in_fingerprints'] and not c['already_in_packs'] else ''
                print(f"  {c['word']:<25} {c['log_odds']:>+8.3f} "
                      f"{c['ai_rate_per_1k']:>7.1f} {c['human_rate_per_1k']:>7.1f} "
                      f"{new:>5}")

        return candidates

    def _get_existing_fingerprints(self):
        """Get current fingerprint words for duplicate detection."""
        try:
            from llm_detector.analyzers.fingerprint import FINGERPRINT_WORDS
            return set(w.lower() for w in FINGERPRINT_WORDS)
        except (ImportError, AttributeError):
            return set()

    def _get_existing_pack_keywords(self):
        """Get all keywords from registered lexicon packs."""
        try:
            from llm_detector.lexicon.packs import PACK_REGISTRY
            all_kw = set()
            for pack in PACK_REGISTRY.values():
                all_kw.update(k.lower() for k in pack.keywords)
            return all_kw
        except (ImportError, AttributeError):
            return set()

    # ── Semantic Centroid Rebuilder (Tool 3) ──────────────────────

    def rebuild_semantic_centroids(self, corpus_path, min_per_class=50):
        """Rebuild semantic centroids from confirmed labeled text.

        Args:
            corpus_path: Path to JSONL with {"task_id": "...", "text": "..."}
            min_per_class: Minimum confirmed examples per class.

        Saves versioned centroids to .beet/centroids/centroids_vXX.npz
        Returns dict with centroid arrays or None.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            print("  Centroid rebuild requires sentence-transformers and numpy")
            return None

        confirmed = {}
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        confirmed[rec['task_id']] = rec['ground_truth']
                    except (json.JSONDecodeError, KeyError):
                        continue

        ai_texts = []
        human_texts = []
        with open(corpus_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    tid = rec.get('task_id', '')
                    text = rec.get('text', rec.get('prompt', ''))
                    if tid in confirmed and text:
                        if confirmed[tid] == 'ai':
                            ai_texts.append(text)
                        elif confirmed[tid] == 'human':
                            human_texts.append(text)
                except json.JSONDecodeError:
                    continue

        if len(ai_texts) < min_per_class or len(human_texts) < min_per_class:
            print(f"  Centroid rebuild: insufficient confirmed text "
                  f"(ai={len(ai_texts)}, human={len(human_texts)}, "
                  f"need >= {min_per_class} each)")
            return None

        print(f"  Computing centroids: {len(ai_texts)} AI, "
              f"{len(human_texts)} human texts...")

        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        ai_embeddings = embedder.encode(ai_texts, show_progress_bar=True)
        human_embeddings = embedder.encode(human_texts, show_progress_bar=True)

        ai_centroid = np.mean(ai_embeddings, axis=0, keepdims=True)
        human_centroid = np.mean(human_embeddings, axis=0, keepdims=True)

        n_clusters = min(5, len(ai_texts) // 20, len(human_texts) // 20)
        n_clusters = max(1, n_clusters)

        if n_clusters > 1:
            from sklearn.cluster import KMeans
            ai_km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            ai_km.fit(ai_embeddings)
            ai_multi_centroids = ai_km.cluster_centers_

            human_km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            human_km.fit(human_embeddings)
            human_multi_centroids = human_km.cluster_centers_
        else:
            ai_multi_centroids = ai_centroid
            human_multi_centroids = human_centroid

        centroids_dir = self.store_dir / 'centroids'
        centroids_dir.mkdir(exist_ok=True)
        version = datetime.now().strftime('%Y%m%d')
        out_path = centroids_dir / f'centroids_v{version}.npz'

        np.savez(
            out_path,
            ai_centroid=ai_centroid,
            human_centroid=human_centroid,
            ai_multi=ai_multi_centroids,
            human_multi=human_multi_centroids,
            n_ai=len(ai_texts),
            n_human=len(human_texts),
            n_clusters=n_clusters,
        )

        import shutil
        latest_path = centroids_dir / 'centroids_latest.npz'
        shutil.copy2(out_path, latest_path)

        print(f"  Centroids saved: {out_path}")
        print(f"    AI: {len(ai_texts)} texts -> {n_clusters} centroid(s)")
        print(f"    Human: {len(human_texts)} texts -> {n_clusters} centroid(s)")

        from sklearn.metrics.pairwise import cosine_similarity
        mean_sep = 1.0 - float(cosine_similarity(ai_centroid, human_centroid)[0][0])
        print(f"    Centroid separation: {mean_sep:.4f} "
              f"(higher = better discrimination)")

        return {
            'ai_centroids': ai_multi_centroids,
            'human_centroids': human_multi_centroids,
            'separation': mean_sep,
            'path': str(out_path),
        }
