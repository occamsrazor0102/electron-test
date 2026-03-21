"""Cross-submission similarity analysis.

v0.65b enhancements:
  FEAT 11 — Occupation-baseline-adjusted (adaptive) thresholds
  FEAT 12 — Semantic similarity via sentence-transformer embeddings
  FEAT 13 — apply_similarity_adjustments() for determination feedback
  FEAT 14 — Cross-batch similarity store (MinHash fingerprints)
  FEAT 15 — Instruction template factoring
"""

import re
import os
import json
import math
import struct
import hashlib
import logging
import statistics
from collections import defaultdict
from llm_detector.compat import HAS_SEMANTIC

logger = logging.getLogger(__name__)


def _word_shingles(text, k=3):
    words = re.findall(r'\w+', text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_STRUCT_FEATURES = [
    'prompt_signature_composite', 'prompt_signature_cfd', 'prompt_signature_mfsr',
    'prompt_signature_must_rate', 'instruction_density_idi',
    'voice_dissonance_spec_score', 'voice_dissonance_voice_score',
    'self_similarity_nssi_score', 'self_similarity_comp_ratio',
    'self_similarity_hapax_ratio', 'self_similarity_sent_length_cv',
    'window_max_score', 'window_mean_score',
    'stylo_fw_ratio', 'stylo_ttr', 'stylo_sent_dispersion',
]


def _structural_similarity(r1, r2):
    diff_sq = sum((r1.get(f, 0) - r2.get(f, 0)) ** 2 for f in _STRUCT_FEATURES)
    return 1.0 / (1.0 + math.sqrt(diff_sq))


# ── FEAT 12: Semantic similarity ──────────────────────────────────────────

def _semantic_similarity(text_a, text_b):
    """Cosine similarity between full-text embeddings."""
    if not HAS_SEMANTIC:
        return 0.0
    from llm_detector.compat import get_semantic_models
    from sklearn.metrics.pairwise import cosine_similarity
    embedder, _, _ = get_semantic_models()
    vecs = embedder.encode([text_a, text_b])
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])


# ── FEAT 14: MinHash fingerprinting ──────────────────────────────────────

def _shingle_fingerprint(shingle_set, n_hashes=128):
    """MinHash fingerprint from shingle set for compact storage."""
    if not shingle_set:
        return [0] * n_hashes
    minhashes = [float('inf')] * n_hashes
    _pack = struct.pack
    _md5 = hashlib.md5
    for shingle in shingle_set:
        shingle_bytes = ' '.join(shingle).encode('utf-8')
        for i in range(n_hashes):
            h = int.from_bytes(_md5(_pack('>I', i) + shingle_bytes).digest()[:4], 'big')
            if h < minhashes[i]:
                minhashes[i] = h
    return minhashes


def _minhash_similarity(fp_a, fp_b):
    """Estimate Jaccard similarity from MinHash fingerprints."""
    if not fp_a or not fp_b or len(fp_a) != len(fp_b):
        return 0.0
    agreement = sum(1 for a, b in zip(fp_a, fp_b) if a == b)
    return agreement / len(fp_a)


# ── FEAT 11 + 12 + 15: Main similarity analysis ─────────────────────────

def analyze_similarity(results, text_map, jaccard_threshold=0.40,
                       struct_threshold=0.90, semantic_threshold=0.92,
                       adaptive=True, instruction_text=None):
    """Analyze cross-submission similarity within occupation groups.

    Args:
        results: List of pipeline result dicts.
        text_map: {task_id: text} mapping.
        jaccard_threshold: Absolute Jaccard threshold (catches copy-paste).
        struct_threshold: Absolute structural similarity threshold.
        semantic_threshold: Absolute semantic similarity threshold (FEAT 12).
        adaptive: If True, compute per-occupation baselines and flag outliers (FEAT 11).
        instruction_text: Optional shared instructions to factor out (FEAT 15).
    """
    by_occ = defaultdict(list)
    for r in results:
        occ = r.get('occupation', '(unknown)')
        by_occ[occ].append(r)

    # FEAT 15: Factor out instruction shingles
    instruction_shingles = set()
    if instruction_text:
        instruction_shingles = _word_shingles(instruction_text)

    shingle_cache = {}
    for tid, text in text_map.items():
        raw_shingles = _word_shingles(text)
        if instruction_shingles:
            shingle_cache[tid] = raw_shingles - instruction_shingles
        else:
            shingle_cache[tid] = raw_shingles

    flagged_pairs = []

    for occ, group in by_occ.items():
        if len(group) < 2:
            continue

        # Phase 1: Compute ALL pairwise similarities within group
        all_jaccards = []
        all_structurals = []
        all_semantics = []
        pair_data = []

        # Pre-compute semantic embeddings for this group if available
        sem_matrix = None
        group_tids = [r.get('task_id', '') for r in group]
        if HAS_SEMANTIC:
            try:
                from llm_detector.compat import get_semantic_models
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                embedder, _, _ = get_semantic_models()
                group_texts = [text_map.get(tid, '') for tid in group_tids]
                if all(t for t in group_texts):
                    group_embeddings = embedder.encode(group_texts)
                    sem_matrix = cos_sim(group_embeddings)
            except Exception as exc:
                logger.debug("Semantic embedding failed for similarity analysis: %s", exc)
                sem_matrix = None

        # Pre-compute attempter names once per group member to avoid
        # repeated .get()/.strip()/.lower() inside the O(n²) inner loop.
        attempters = [r.get('attempter', '').strip().lower() for r in group]

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                r_a, r_b = group[i], group[j]
                att_a = attempters[i]
                att_b = attempters[j]

                tid_a = r_a.get('task_id', '')
                tid_b = r_b.get('task_id', '')

                jac = _jaccard(
                    shingle_cache.get(tid_a, set()),
                    shingle_cache.get(tid_b, set()),
                )
                struct = _structural_similarity(r_a, r_b)

                sem = 0.0
                if sem_matrix is not None:
                    sem = float(sem_matrix[i][j])

                all_jaccards.append(jac)
                all_structurals.append(struct)
                all_semantics.append(sem)
                pair_data.append((i, j, jac, struct, sem, att_a, att_b, tid_a, tid_b))

        # Phase 2: Compute occupation baseline (FEAT 11)
        jac_median = jac_std = 0.0
        struct_median = struct_std = 0.0
        sem_median = sem_std = 0.0
        adaptive_jac_threshold = jaccard_threshold
        adaptive_struct_threshold = struct_threshold
        adaptive_sem_threshold = semantic_threshold

        if adaptive and len(all_jaccards) >= 3:
            jac_median = statistics.median(all_jaccards)
            jac_std = statistics.stdev(all_jaccards)
            struct_median = statistics.median(all_structurals)
            struct_std = statistics.stdev(all_structurals)

            adaptive_jac_threshold = jac_median + 2.0 * max(jac_std, 0.02)
            adaptive_struct_threshold = struct_median + 2.0 * max(struct_std, 0.02)

            if all_semantics and any(s > 0 for s in all_semantics):
                sem_median = statistics.median(all_semantics)
                sem_std = statistics.stdev(all_semantics) if len(all_semantics) >= 3 else 0.02
                adaptive_sem_threshold = sem_median + 2.0 * max(sem_std, 0.01)

        # Phase 3: Flag pairs exceeding thresholds
        for i, j, jac, struct, sem, att_a, att_b, tid_a, tid_b in pair_data:
            r_a, r_b = group[i], group[j]

            if att_a and att_b and att_a == att_b:
                continue

            flags = []
            flag_details = {}

            # Absolute threshold (catches copy-paste)
            if jac >= jaccard_threshold:
                flags.append('text')

            # Adaptive threshold (catches same-template generation)
            if adaptive and jac >= adaptive_jac_threshold and jac > jac_median + 0.05:
                if 'text' not in flags:
                    flags.append('text_adaptive')
                flag_details['jac_z'] = round(
                    (jac - jac_median) / max(jac_std, 0.01), 2
                )

            # Structural similarity
            if struct >= struct_threshold:
                flags.append('structural')
            if adaptive and struct >= adaptive_struct_threshold and struct > struct_median + 0.03:
                if 'structural' not in flags:
                    flags.append('structural_adaptive')
                flag_details['struct_z'] = round(
                    (struct - struct_median) / max(struct_std, 0.01), 2
                )

            # Semantic similarity (FEAT 12)
            if sem >= semantic_threshold:
                flags.append('semantic')
            if adaptive and sem > 0 and sem >= adaptive_sem_threshold and sem > sem_median + 0.03:
                if 'semantic' not in flags:
                    flags.append('semantic_adaptive')
                flag_details['sem_z'] = round(
                    (sem - sem_median) / max(sem_std, 0.01), 2
                )

            if flags:
                pair_dict = {
                    'id_a': tid_a,
                    'id_b': tid_b,
                    'attempter_a': r_a.get('attempter', ''),
                    'attempter_b': r_b.get('attempter', ''),
                    'occupation': occ,
                    'jaccard': round(jac, 4),
                    'structural': round(struct, 4),
                    'semantic': round(sem, 4),
                    'flag_type': '+'.join(flags),
                    'det_a': r_a['determination'],
                    'det_b': r_b['determination'],
                    'occ_jac_median': round(jac_median, 4),
                    'occ_jac_std': round(jac_std, 4) if jac_std else 0.0,
                    'occ_struct_median': round(struct_median, 4),
                }
                pair_dict.update(flag_details)
                flagged_pairs.append(pair_dict)

        # Emit baseline stats for this occupation group
        if len(all_jaccards) >= 3:
            flagged_pairs.append({
                '_type': 'baseline',
                'occupation': occ,
                'n_pairs': len(all_jaccards),
                'jac_median': round(jac_median, 4),
                'jac_std': round(jac_std, 4),
                'jac_p90': round(statistics.quantiles(all_jaccards, n=10)[8], 4),
                'struct_median': round(struct_median, 4),
                'struct_std': round(struct_std, 4),
                'sem_median': round(sem_median, 4),
                'adaptive_jac_threshold': round(adaptive_jac_threshold, 4),
                'adaptive_struct_threshold': round(adaptive_struct_threshold, 4),
                'adaptive_sem_threshold': round(adaptive_sem_threshold, 4),
            })

    flagged_pairs.sort(key=lambda p: p.get('jaccard', 0), reverse=True)
    return flagged_pairs


# ── FEAT 13: Similarity feedback into determination ──────────────────────

def apply_similarity_adjustments(results, sim_pairs, text_map):
    """Adjust determinations based on similarity findings.

    Rules:
    - If a prompt is in a flagged similarity pair AND its current
      determination is YELLOW or GREEN, upgrade to the next level.
    - If a prompt has 2+ similarity flags with different partners,
      upgrade more aggressively (indicates systematic template use).
    - Never downgrade a determination based on similarity.
    - Add similarity context to the result's audit trail.

    Returns modified results list.
    """
    sim_profile = defaultdict(list)
    for p in sim_pairs:
        if p.get('_type') == 'baseline':
            continue
        sim_profile[p['id_a']].append(p)
        sim_profile[p['id_b']].append(p)

    upgrade_map = {'GREEN': 'YELLOW', 'REVIEW': 'YELLOW', 'YELLOW': 'AMBER'}

    for r in results:
        tid = r.get('task_id', '')
        pairs = sim_profile.get(tid, [])
        if not pairs:
            continue

        n_partners = len(set(
            p['id_b'] if p['id_a'] == tid else p['id_a']
            for p in pairs
        ))

        max_jac = max(p.get('jaccard', 0) for p in pairs)
        max_sem = max(p.get('semantic', 0) for p in pairs)
        has_semantic = any('semantic' in p.get('flag_type', '') for p in pairs)
        has_adaptive = any('adaptive' in p.get('flag_type', '') for p in pairs)

        current_det = r['determination']
        new_det = current_det
        sim_reason = None

        if has_semantic and n_partners >= 2:
            if current_det in upgrade_map:
                new_det = 'AMBER'
                sim_reason = (f"Semantic similarity with {n_partners} other submissions "
                             f"(max={max_sem:.2f}) suggests shared LLM template")
        elif has_semantic and current_det in ('GREEN', 'REVIEW', 'YELLOW'):
            new_det = upgrade_map.get(current_det, current_det)
            sim_reason = (f"High semantic similarity (max={max_sem:.2f}) with another "
                         f"submission from a different attempter")
        elif has_adaptive and current_det in ('GREEN', 'REVIEW'):
            new_det = 'YELLOW'
            sim_reason = (f"Similarity exceeds occupation baseline "
                         f"(Jaccard={max_jac:.2f}, z-score in pair data)")

        if new_det != current_det:
            r['determination'] = new_det
            r['similarity_upgrade'] = {
                'original_determination': current_det,
                'upgraded_to': new_det,
                'reason': sim_reason,
                'n_similar_partners': n_partners,
                'max_jaccard': round(max_jac, 4),
                'max_semantic': round(max_sem, 4),
            }
            r['reason'] = f"{r['reason']} + SIM: {sim_reason}"

        r['similarity_partners'] = n_partners
        r['similarity_max_jaccard'] = round(max_jac, 4)
        r['similarity_max_semantic'] = round(max_sem, 4)

    return results


# ── FEAT 14: Cross-batch similarity store ────────────────────────────────

def save_similarity_store(results, text_map, store_path):
    """Append fingerprints from current batch to persistent store.

    Stores MinHash fingerprints and structural features — NOT full text.
    """
    import datetime
    records = []
    for r in results:
        tid = r.get('task_id', '')
        text = text_map.get(tid, '')
        if not text:
            continue

        shingles = _word_shingles(text)
        minhash = _shingle_fingerprint(shingles)
        struct_vec = {f: r.get(f, 0) for f in _STRUCT_FEATURES}

        records.append({
            'task_id': tid,
            'attempter': r.get('attempter', ''),
            'occupation': r.get('occupation', ''),
            'determination': r['determination'],
            'minhash': minhash,
            'structural': struct_vec,
            'word_count': r.get('word_count', 0),
            '_batch_timestamp': datetime.datetime.now().isoformat(),
        })

    with open(store_path, 'a') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"  Similarity store: {len(records)} fingerprints appended to {store_path}")


def load_similarity_store(store_path):
    """Load previously stored fingerprints."""
    if not os.path.exists(store_path):
        return []

    records = []
    with open(store_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def cross_batch_similarity(current_results, text_map, store_path,
                           minhash_threshold=0.50):
    """Compare current batch against previously stored fingerprints.

    Returns list of cross-batch similarity flags.
    """
    historical = load_similarity_store(store_path)
    if not historical:
        return []

    flags = []

    for r in current_results:
        tid = r.get('task_id', '')
        text = text_map.get(tid, '')
        if not text:
            continue

        current_shingles = _word_shingles(text)
        current_minhash = _shingle_fingerprint(current_shingles)
        # Compute once per outer iteration — att_curr doesn't depend on hist.
        att_curr = r.get('attempter', '').strip().lower()

        for hist in historical:
            if hist['task_id'] == tid:
                continue

            att_hist = hist.get('attempter', '').strip().lower()
            if att_curr and att_hist and att_curr == att_hist:
                continue

            mh_sim = _minhash_similarity(current_minhash, hist.get('minhash', []))

            if mh_sim >= minhash_threshold:
                struct = _structural_similarity(r, hist.get('structural', {}))

                flags.append({
                    'current_id': tid,
                    'historical_id': hist['task_id'],
                    'current_attempter': r.get('attempter', ''),
                    'historical_attempter': hist.get('attempter', ''),
                    'occupation': r.get('occupation', ''),
                    'minhash_similarity': round(mh_sim, 3),
                    'structural_similarity': round(struct, 4),
                    'historical_determination': hist.get('determination', '?'),
                    'historical_batch': hist.get('_batch_timestamp', '?'),
                })

    flags.sort(key=lambda f: f['minhash_similarity'], reverse=True)
    return flags


# ── Print helpers ────────────────────────────────────────────────────────

def print_similarity_report(pairs):
    """Print cross-submission similarity findings."""
    actual_pairs = [p for p in pairs if p.get('_type') != 'baseline']
    baselines = [p for p in pairs if p.get('_type') == 'baseline']

    if not actual_pairs:
        print("\n  No cross-attempter similarity clusters detected.")
        if baselines:
            for b in baselines:
                print(f"    Baseline [{b['occupation'][:30]}]: "
                      f"Jac median={b['jac_median']:.3f} (n={b['n_pairs']} pairs)")
        return

    print(f"\n{'='*90}")
    print(f"  SIMILARITY CLUSTERS: {len(actual_pairs)} flagged pairs")
    print(f"{'='*90}")

    for b in baselines:
        print(f"\n  Baseline [{b['occupation'][:30]}]: "
              f"Jac median={b['jac_median']:.3f} p90={b['jac_p90']:.3f} "
              f"sem median={b.get('sem_median', 0):.3f} "
              f"(n={b['n_pairs']} pairs)")

    for p in actual_pairs:
        icon = 'RED' if p['jaccard'] >= 0.70 else 'AMBER' if p['jaccard'] >= 0.50 else 'YELLOW'
        sem_str = f"  Sem={p.get('semantic', 0):.2f}" if p.get('semantic', 0) > 0 else ""
        print(f"\n  [{icon}] Jac={p['jaccard']:.2f}  Struct={p['structural']:.2f}{sem_str}  [{p['flag_type']}]")
        print(f"     {p['id_a'][:15]:15s} ({p['attempter_a'] or '?':20s}) [{p['det_a']}]")
        print(f"     {p['id_b'][:15]:15s} ({p['attempter_b'] or '?':20s}) [{p['det_b']}]")
        print(f"     Occupation: {p['occupation'][:50]}")
