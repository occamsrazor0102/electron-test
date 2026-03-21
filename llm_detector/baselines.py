"""Labeled data collection and baseline analysis."""

import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

_BASELINE_FIELDS = [
    'task_id', 'occupation', 'attempter', 'word_count', 'determination',
    'confidence', 'preamble_score', 'prompt_signature_composite', 'prompt_signature_cfd',
    'prompt_signature_mfsr', 'prompt_signature_framing', 'prompt_signature_must_rate',
    'prompt_signature_distinct_frames',
    'instruction_density_idi', 'instruction_density_imperatives', 'instruction_density_conditionals',
    'voice_dissonance_voice_score', 'voice_dissonance_spec_score', 'voice_dissonance_vsd',
    'voice_dissonance_voice_gated', 'voice_dissonance_hedges', 'voice_dissonance_casual_markers',
    'voice_dissonance_misspellings', 'ssi_triggered',
    'self_similarity_nssi_score', 'self_similarity_nssi_signals', 'self_similarity_determination',
    'continuation_bscore', 'continuation_determination',
    'self_similarity_sent_length_cv', 'self_similarity_comp_ratio', 'self_similarity_hapax_ratio',
    'norm_obfuscation_delta', 'norm_invisible_chars', 'norm_homoglyphs', 'norm_attack_types',
    'attack_type',
    'lang_support_level', 'lang_fw_coverage', 'lang_non_latin_ratio',
    'ground_truth', 'language', 'domain', 'mode',
    'window_max_score', 'window_mean_score', 'window_variance',
    'window_hot_span', 'window_mixed_signal',
    'stylo_fw_ratio', 'stylo_sent_dispersion', 'stylo_ttr',
    'calibrated_confidence', 'conformity_level', 'calibration_stratum',
    'pack_constraint_score', 'pack_exec_spec_score', 'pack_schema_score',
    'pack_active_families', 'pack_prompt_boost', 'pack_idi_boost',
    'perplexity_value', 'surprisal_variance', 'volatility_decay_ratio',
    'continuation_composite_stability', 'continuation_composite_variance',
    'continuation_improvement_rate', 'continuation_ncd_matrix_variance',
    'window_fw_trajectory_cv', 'window_comp_trajectory_cv',
    'tocsin_cohesiveness', 'perplexity_zlib_normalized_ppl',
    'self_similarity_structural_compression_delta',
    'surprisal_trajectory_cv', 'surprisal_stationarity',
    'binoculars_score',
]


def derive_attack_type(record):
    """Derive categorical attack_type from normalization fields."""
    homoglyphs = record.get('norm_homoglyphs', 0) or 0
    invisible = record.get('norm_invisible_chars', 0) or 0
    delta = record.get('norm_obfuscation_delta', 0) or 0

    tags = []
    if homoglyphs > 0:
        tags.append('homoglyph')
    if invisible > 0:
        tags.append('zero_width')
    if delta >= 0.02 and not tags:
        tags.append('encoding')

    if len(tags) > 1:
        return 'combined'
    elif len(tags) == 1:
        return tags[0]
    return 'none'


def collect_baselines(results, output_path):
    """Append scored results to JSONL file for baseline accumulation."""
    timestamp = datetime.now().isoformat()
    n_written = 0

    with open(output_path, 'a') as f:
        for r in results:
            record = {k: r.get(k) for k in _BASELINE_FIELDS}
            record['attack_type'] = derive_attack_type(record)
            record['_timestamp'] = timestamp
            record['_version'] = 'v0.66'
            wc = r.get('word_count', 0)
            if wc < 100:
                record['length_bin'] = 'short'
            elif wc < 300:
                record['length_bin'] = 'medium'
            elif wc < 800:
                record['length_bin'] = 'long'
            else:
                record['length_bin'] = 'very_long'
            f.write(json.dumps(record) + '\n')
            n_written += 1

    print(f"\n  Baseline data: {n_written} records appended to {output_path}")
    return n_written


def analyze_baselines(jsonl_path, output_csv=None):
    """Read accumulated baseline JSONL and compute per-occupation percentile tables."""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print(f"No records found in {jsonl_path}")
        return

    df = pd.DataFrame(records)
    print(f"\n{'='*90}")
    print(f"  BASELINE ANALYSIS -- {len(df)} records from {jsonl_path}")
    print(f"{'='*90}")

    det_counts = df['determination'].value_counts()
    print(f"\n  Overall distribution:")
    for det in ['RED', 'AMBER', 'YELLOW', 'GREEN']:
        ct = det_counts.get(det, 0)
        pct = ct / len(df) * 100
        print(f"    {det:>8}: {ct:>5} ({pct:.1f}%)")

    metrics = ['prompt_signature_composite', 'prompt_signature_cfd', 'prompt_signature_mfsr',
               'prompt_signature_must_rate', 'instruction_density_idi',
               'voice_dissonance_spec_score', 'voice_dissonance_voice_score', 'voice_dissonance_vsd',
               'self_similarity_nssi_score', 'self_similarity_comp_ratio', 'self_similarity_hapax_ratio',
               'self_similarity_sent_length_cv',
               'norm_obfuscation_delta', 'lang_fw_coverage', 'word_count']
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    occupations = sorted(df['occupation'].dropna().unique())
    if not occupations:
        occupations = ['(all)']
        df['occupation'] = '(all)'

    all_rows = []

    for occ in occupations:
        occ_df = df[df['occupation'] == occ]
        if len(occ_df) < 5:
            continue

        print(f"\n  -- {occ} (n={len(occ_df)}) --")
        det_pcts = occ_df['determination'].value_counts()
        flags = det_pcts.get('RED', 0) + det_pcts.get('AMBER', 0)
        flag_rate = flags / len(occ_df) * 100
        print(f"     Flag rate: {flag_rate:.1f}% ({flags}/{len(occ_df)})")

        for m in metrics:
            if m not in occ_df.columns:
                continue
            vals = pd.to_numeric(occ_df[m], errors='coerce').dropna()
            if len(vals) < 3:
                continue

            pct_vals = vals.quantile(percentiles).to_dict()
            row = {'occupation': occ, 'metric': m, 'n': len(vals),
                   'mean': vals.mean(), 'std': vals.std()}
            row.update({f'p{int(k*100)}': v for k, v in pct_vals.items()})
            all_rows.append(row)

            p50 = pct_vals.get(0.50, 0)
            p90 = pct_vals.get(0.90, 0)
            p99 = pct_vals.get(0.99, 0)
            print(f"     {m:40s}  p50={p50:7.2f}  p90={p90:7.2f}  p99={p99:7.2f}  mean={vals.mean():7.2f}")

    if output_csv and all_rows:
        baseline_df = pd.DataFrame(all_rows)
        baseline_df.to_csv(output_csv, index=False)
        print(f"\n  Baseline percentiles written to {output_csv}")

    if 'ground_truth' in df.columns:
        labeled = df[df['ground_truth'].isin(['human', 'ai'])].copy()
        if len(labeled) >= 20:
            n_human = (labeled['ground_truth'] == 'human').sum()
            n_ai = (labeled['ground_truth'] == 'ai').sum()
            print(f"\n  -- TPR @ FPR (n={len(labeled)}: {n_human} human, {n_ai} AI) --")

            if n_human >= 5 and n_ai >= 5:
                scores = pd.to_numeric(labeled['confidence'], errors='coerce').fillna(0)
                labels = (labeled['ground_truth'] == 'ai').astype(int)

                thresholds = sorted(scores.unique(), reverse=True)
                for target_fpr, label in [(0.01, '1%'), (0.05, '5%'), (0.10, '10%')]:
                    best_tpr = 0.0
                    best_thresh = 1.0
                    for t in thresholds:
                        predicted_pos = (scores >= t)
                        fp = ((predicted_pos) & (labels == 0)).sum()
                        tp = ((predicted_pos) & (labels == 1)).sum()
                        fpr = fp / max(n_human, 1)
                        tpr = tp / max(n_ai, 1)
                        if fpr <= target_fpr and tpr > best_tpr:
                            best_tpr = tpr
                            best_thresh = t
                    print(f"     TPR @ {label:>3} FPR: {best_tpr:.1%}  (threshold={best_thresh:.3f})")

                for gt_label in ['human', 'ai']:
                    subset = labeled[labeled['ground_truth'] == gt_label]
                    flagged = subset['determination'].isin(['RED', 'AMBER']).sum()
                    rate = flagged / max(len(subset), 1) * 100
                    print(f"     Flag rate ({gt_label:>5}): {rate:.1f}% ({flagged}/{len(subset)})")

    if 'domain' in df.columns and 'length_bin' in df.columns:
        df['_stratum'] = df['domain'].fillna('unknown').astype(str) + 'x' + df['length_bin'].fillna('unknown').astype(str)
        strata = df['_stratum'].unique()
        if len(strata) > 1:
            print(f"\n  -- STRATIFIED FLAG RATES (domain x length_bin) --")
            stratum_rates = {}
            for s in sorted(strata):
                s_df = df[df['_stratum'] == s]
                if len(s_df) < 3:
                    continue
                flagged = s_df['determination'].isin(['RED', 'AMBER']).sum()
                rate = flagged / len(s_df) * 100
                stratum_rates[s] = rate
                print(f"     {s:30s}  n={len(s_df):>4}  flag_rate={rate:5.1f}%")

            if stratum_rates:
                rates = list(stratum_rates.values())
                max_rate = max(rates)
                min_rate = min(rates)
                if max_rate - min_rate > 20:
                    print(f"\n  WARNING: Flag rate disparity across strata")
                    print(f"     Range: {min_rate:.1f}% -- {max_rate:.1f}% (delta={max_rate - min_rate:.1f}pp)")

    return all_rows
