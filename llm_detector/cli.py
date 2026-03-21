"""Command-line interface for the LLM Detection Pipeline."""

import json
import os
import argparse
import subprocess
import importlib.util
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import sys

import pandas as pd

from llm_detector.pipeline import analyze_prompt
from llm_detector.calibration import (
    calibrate_from_baselines, save_calibration, load_calibration,
)
from llm_detector.baselines import analyze_baselines, collect_baselines
from llm_detector.similarity import (
    analyze_similarity, print_similarity_report,
    apply_similarity_adjustments, save_similarity_store, cross_batch_similarity,
)
from llm_detector.io import load_xlsx, load_csv, load_pdf
from llm_detector._constants import STREAMLIT_MIN_VERSION as _STREAMLIT_MIN_VERSION


def _is_frozen():
    """Check if running as a PyInstaller bundle."""
    return getattr(sys, 'frozen', False)


def _real_python():
    """Return the real Python interpreter, even inside a frozen PyInstaller bundle."""
    if _is_frozen():
        for name in ('python3', 'python'):
            path = shutil.which(name)
            if path:
                return path
        for candidate in ('/usr/bin/python3', '/usr/local/bin/python3'):
            if os.path.isfile(candidate):
                return candidate
    return sys.executable


def print_result(r, verbose=False):
    """Pretty-print a single result."""
    icons = {'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1',
             'GREEN': '\U0001f7e2', 'MIXED': '\U0001f535', 'REVIEW': '\u26aa'}
    icon = icons.get(r['determination'], '?')

    print(f"\n  {icon} [{r['determination']}] {r['task_id'][:20]}  |  {r['occupation'][:45]}")
    print(f"     Attempter: {r['attempter'] or '(unknown)'} | Stage: {r['stage']} | Words: {r['word_count']} | Mode: {r.get('mode', '?')}")
    print(f"     Reason: {r['reason']}")

    cal_conf = r.get('calibrated_confidence')
    p_val = r.get('conformity_level')
    if cal_conf is not None and cal_conf != r.get('confidence'):
        cal_str = f"     Calibrated: conf={cal_conf:.3f}"
        if p_val is not None:
            cal_str += f"  conf_level={p_val:.3f}"
        cal_str += f"  [{r.get('calibration_stratum', '?')}]"
        print(cal_str)

    shadow = r.get('shadow_disagreement')
    if shadow:
        print(f"     \u26a0\ufe0f SHADOW: {shadow['interpretation']}")
        print(f"         Rule={shadow['rule_determination']}, "
              f"Model={shadow['shadow_ai_prob']:.1%} AI")

    if verbose or r['determination'] in ('RED', 'AMBER'):
        delta = r.get('norm_obfuscation_delta', 0)
        lang = r.get('lang_support_level', 'SUPPORTED')
        if delta > 0 or lang != 'SUPPORTED':
            print(f"     NORM obfuscation: {delta:.1%}  invisible={r.get('norm_invisible_chars', 0)} homoglyphs={r.get('norm_homoglyphs', 0)}")
            print(f"     GATE support:     {lang} (fw_coverage={r.get('lang_fw_coverage', 0):.2f}, non_latin={r.get('lang_non_latin_ratio', 0):.2f})")
        print(f"     Preamble:         {r['preamble_score']:.2f} ({r['preamble_severity']}, {r['preamble_hits']} hits)")
        if r['preamble_details']:
            for name, sev in r['preamble_details']:
                print(f"         -> [{sev}] {name}")
        print(f"     Fingerprints:     {r['fingerprint_score']:.2f} ({r['fingerprint_hits']} hits)")
        print(f"     Prompt Sig:       {r['prompt_signature_composite']:.2f}")
        print(f"         CFD={r['prompt_signature_cfd']:.3f} frames={r['prompt_signature_distinct_frames']} MFSR={r['prompt_signature_mfsr']:.3f}")
        print(f"         meta={r['prompt_signature_meta_design']} FC={r['prompt_signature_framing']}/3 must={r['prompt_signature_must_rate']:.3f}/sent")
        print(f"         contractions={r['prompt_signature_contractions']} numbered_criteria={r['prompt_signature_numbered_criteria']}")
        print(f"     IDI:              {r['instruction_density_idi']:.1f}  (imp={r['instruction_density_imperatives']} cond={r['instruction_density_conditionals']} Y/N={r['instruction_density_binary_specs']} MISS={r['instruction_density_missing_refs']} flag={r['instruction_density_flag_count']})")
        print(f"     VSD:              {r['voice_dissonance_vsd']:.1f}  (voice={r['voice_dissonance_voice_score']:.1f} x spec={r['voice_dissonance_spec_score']:.1f})")
        print(f"         gated={'YES' if r['voice_dissonance_voice_gated'] else 'no'} casual={r['voice_dissonance_casual_markers']} typos={r['voice_dissonance_misspellings']}")
        print(f"         cols={r['voice_dissonance_camel_cols']} calcs={r['voice_dissonance_calcs']} hedges={r['voice_dissonance_hedges']}")
        if r.get('ssi_triggered'):
            print(f"     SSI:  TRIGGERED  (spec={r['voice_dissonance_spec_score']:.1f}, voice=0, hedges=0, {r['word_count']}w)")
        nssi_score = r.get('self_similarity_nssi_score', 0.0)
        nssi_signals = r.get('self_similarity_nssi_signals', 0)
        nssi_det = r.get('self_similarity_determination')
        if nssi_score > 0 or nssi_det:
            det_str = nssi_det or 'n/a'
            print(f"     NSSI:             {nssi_score:.3f}  ({nssi_signals} signals, det={det_str})")
            print(f"         formulaic={r.get('self_similarity_formulaic_density', 0):.3f} power_adj={r.get('self_similarity_power_adj_density', 0):.3f}"
                  f" demo={r.get('self_similarity_demonstrative_density', 0):.3f} trans={r.get('self_similarity_transition_density', 0):.3f}")
            print(f"         sent_cv={r.get('self_similarity_sent_length_cv', 0):.3f} comp_ratio={r.get('self_similarity_comp_ratio', 0):.3f}"
                  f" hapax={r.get('self_similarity_hapax_ratio', 0):.3f} (unique={r.get('self_similarity_unique_words', 0)})")
        bscore = r.get('continuation_bscore', 0.0)
        dna_det = r.get('continuation_determination')
        if bscore > 0 or dna_det:
            det_str = dna_det or 'n/a'
            print(f"     DNA-GPT:          BScore={bscore:.4f}  (max={r.get('continuation_bscore_max', 0):.4f}, "
                  f"samples={r.get('continuation_n_samples', 0)}, det={det_str})")

        shadow = r.get('shadow_disagreement')
        if shadow:
            print(f"     \u26a0\ufe0f SHADOW: {shadow['interpretation']}")
            print(f"         Rule={shadow['rule_determination']}, "
                  f"Model={shadow['shadow_ai_prob']:.1%} AI")

        cd = r.get('channel_details', {})
        if cd.get('channels'):
            print(f"     -- Channels --")
            for ch_name, ch_info in cd['channels'].items():
                if ch_info['severity'] != 'GREEN':
                    eligible = 'Y' if ch_info.get('mode_eligible') else 'o'
                    role = ch_info.get('role', '')
                    role_tag = f'[{role}] ' if role else ''
                    print(f"     {eligible} {ch_name:18s} {ch_info['severity']:6s} score={ch_info['score']:.2f}  {role_tag}{ch_info['explanation'][:60]}")
            triggering_rule = cd.get('triggering_rule', '')
            if triggering_rule:
                print(f"     rule: {triggering_rule}")


# ==============================================================================
# INTERACTIVE LABELING
# ==============================================================================


def _sort_for_labeling(results):
    """Sort results to prioritize cases where human labels are most valuable.

    Order: YELLOW first (most ambiguous), then AMBER, then MIXED,
    then RED (confirm true positives), then GREEN (confirm true negatives).
    Within each tier, lower confidence first (harder calls first while fresh).
    """
    tier_order = {'YELLOW': 0, 'MIXED': 1, 'AMBER': 2, 'RED': 3, 'REVIEW': 4, 'GREEN': 5}
    return sorted(results, key=lambda r: (
        tier_order.get(r.get('determination', 'GREEN'), 6),
        r.get('confidence', 0),
    ))


def _format_labeling_display(r, text_map=None, show_text_chars=300):
    """Format a single result for the labeling interface."""
    icons = {'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1',
             'GREEN': '\U0001f7e2', 'MIXED': '\U0001f535', 'REVIEW': '\u26aa'}
    icon = icons.get(r['determination'], '?')

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {icon} [{r['determination']}]  conf={r.get('confidence', 0):.2f}  "
                 f"mode={r.get('mode', '?')}")
    lines.append(f"  Task:      {r.get('task_id', '?')}")
    lines.append(f"  Attempter: {r.get('attempter', '(unknown)')}")
    lines.append(f"  Occupation:{r.get('occupation', '(unknown)')}")
    lines.append(f"  Words:     {r.get('word_count', 0)}")
    lines.append(f"  Reason:    {r.get('reason', '')[:120]}")

    # Key signals summary
    lines.append(f"\n  --- Key Signals ---")
    lines.append(f"  Preamble:    {r.get('preamble_score', 0):.2f} ({r.get('preamble_severity', 'NONE')})")
    lines.append(f"  Prompt Sig:  {r.get('prompt_signature_composite', 0):.2f} "
                 f"(CFD={r.get('prompt_signature_cfd', 0):.3f})")
    lines.append(f"  VSD:         {r.get('voice_dissonance_vsd', 0):.1f} "
                 f"(voice={r.get('voice_dissonance_voice_score', 0):.1f} x "
                 f"spec={r.get('voice_dissonance_spec_score', 0):.1f})")
    lines.append(f"  IDI:         {r.get('instruction_density_idi', 0):.1f}")
    lines.append(f"  NSSI:        {r.get('self_similarity_nssi_score', 0):.3f} "
                 f"({r.get('self_similarity_nssi_signals', 0)} signals)")
    lines.append(f"  DNA-GPT:     {r.get('continuation_bscore', 0):.4f} "
                 f"({r.get('continuation_mode', 'n/a')})")

    cd = r.get('channel_details', {})
    channels_cd = cd.get('channels', {})
    if channels_cd:
        lines.append(f"\n  --- Channels ---")
        for ch_name in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
            info = channels_cd.get(ch_name, {})
            sev = info.get('severity', 'GREEN')
            if sev != 'GREEN':
                role = info.get('role', '')
                role_tag = f' [{role}]' if role else ''
                lines.append(f"  {ch_name:20s} {sev:6s}{role_tag}  {info.get('explanation', '')[:60]}")
        triggering_rule = cd.get('triggering_rule', '')
        if triggering_rule:
            lines.append(f"  rule: {triggering_rule}")

    # Text preview
    if text_map and r.get('task_id') in text_map:
        text = text_map[r['task_id']]
        preview = text[:show_text_chars]
        if len(text) > show_text_chars:
            preview += f"... [{len(text) - show_text_chars} more chars]"
        lines.append(f"\n  --- Text Preview ---")
        lines.append(f"  {preview}")

    lines.append(f"{'='*80}")
    return '\n'.join(lines)


def interactive_label(results, text_map=None, output_path=None, reviewer='',
                      store=None, skip_green=False, skip_red=False,
                      max_labels=None):
    """Interactive labeling session for calibration data collection.

    Presents each result to the reviewer and collects ground truth labels.
    Saves labels to JSONL for calibration and optionally to a MemoryStore.

    Args:
        results: List of pipeline result dicts.
        text_map: Optional dict mapping task_id -> original text.
        output_path: JSONL path to append labeled records. If None, uses
                     'beet_labels_{date}.jsonl'.
        reviewer: Reviewer identifier string.
        store: Optional MemoryStore for record_confirmation integration.
        skip_green: Skip GREEN determinations (assume correct).
        skip_red: Skip RED determinations (assume correct).
        max_labels: Stop after this many labels (None = label all).

    Returns:
        dict with labeling session statistics.
    """
    if output_path is None:
        output_path = f"beet_labels_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"

    sorted_results = _sort_for_labeling(results)

    if skip_green:
        sorted_results = [r for r in sorted_results if r['determination'] != 'GREEN']
    if skip_red:
        sorted_results = [r for r in sorted_results if r['determination'] != 'RED']

    if max_labels:
        sorted_results = sorted_results[:max_labels]

    if not reviewer:
        reviewer = input("  Reviewer name/ID: ").strip() or 'anonymous'

    stats = {
        'total_presented': 0,
        'labeled_ai': 0,
        'labeled_human': 0,
        'labeled_unsure': 0,
        'skipped': 0,
        'true_positives': 0,   # pipeline flagged + human says AI
        'false_positives': 0,  # pipeline flagged + human says human
        'true_negatives': 0,   # pipeline clean + human says human
        'false_negatives': 0,  # pipeline clean + human says AI
        'reviewer': reviewer,
    }

    print(f"\n{'#'*80}")
    print(f"  BEET INTERACTIVE LABELING SESSION")
    print(f"  {len(sorted_results)} items to review | Reviewer: {reviewer}")
    print(f"  Labels: (a)i  (h)uman  (u)nsure  (s)kip  (q)uit")
    print(f"  Optional notes: type after label, e.g. 'a obvious template'")
    print(f"{'#'*80}")

    labeled_records = []

    for i, r in enumerate(sorted_results):
        stats['total_presented'] += 1

        print(_format_labeling_display(r, text_map))
        print(f"\n  [{i+1}/{len(sorted_results)}] Label this text:")

        try:
            raw = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Session interrupted.")
            break

        if not raw:
            stats['skipped'] += 1
            continue

        # Parse input: first char is label, rest is notes
        label_char = raw[0].lower()
        notes = raw[1:].strip() if len(raw) > 1 else ''

        if label_char == 'q':
            print("  Session ended by reviewer.")
            break
        elif label_char == 's':
            stats['skipped'] += 1
            continue
        elif label_char == 'a':
            ground_truth = 'ai'
            stats['labeled_ai'] += 1
        elif label_char == 'h':
            ground_truth = 'human'
            stats['labeled_human'] += 1
        elif label_char == 'u':
            ground_truth = 'unsure'
            stats['labeled_unsure'] += 1
        else:
            print(f"  Unknown label '{label_char}' — skipping.")
            stats['skipped'] += 1
            continue

        # Confusion matrix tracking
        pipeline_flagged = r.get('determination') in ('RED', 'AMBER', 'MIXED')
        if ground_truth == 'ai' and pipeline_flagged:
            stats['true_positives'] += 1
        elif ground_truth == 'human' and pipeline_flagged:
            stats['false_positives'] += 1
        elif ground_truth == 'human' and not pipeline_flagged:
            stats['true_negatives'] += 1
        elif ground_truth == 'ai' and not pipeline_flagged:
            stats['false_negatives'] += 1

        # Build labeled record
        record = {
            'task_id': r.get('task_id', ''),
            'attempter': r.get('attempter', ''),
            'occupation': r.get('occupation', ''),
            'ground_truth': ground_truth,
            'pipeline_determination': r.get('determination', ''),
            'pipeline_confidence': r.get('confidence', 0),
            'reviewer': reviewer,
            'notes': notes,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': 'v0.66',
            # Carry forward all scores for calibration
            'confidence': r.get('confidence', 0),
            'word_count': r.get('word_count', 0),
            'domain': r.get('domain', ''),
            'mode': r.get('mode', ''),
        }

        # Length bin for stratified calibration
        wc = r.get('word_count', 0)
        if wc < 100:
            record['length_bin'] = 'short'
        elif wc < 300:
            record['length_bin'] = 'medium'
        elif wc < 800:
            record['length_bin'] = 'long'
        else:
            record['length_bin'] = 'very_long'

        labeled_records.append(record)

        # Write to JSONL immediately (crash-safe)
        with open(output_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Write to memory store if available
        if store and ground_truth in ('ai', 'human'):
            store.record_confirmation(
                r.get('task_id', ''), ground_truth,
                verified_by=reviewer, notes=notes,
            )

        print(f"  Recorded: {ground_truth}" + (f" ({notes})" if notes else ""))

    # Session summary
    _print_labeling_summary(stats, output_path)

    return stats


def _print_labeling_summary(stats, output_path):
    """Print end-of-session summary with calibration readiness check."""
    total_labeled = stats['labeled_ai'] + stats['labeled_human'] + stats['labeled_unsure']

    print(f"\n{'#'*80}")
    print(f"  LABELING SESSION COMPLETE")
    print(f"{'#'*80}")
    print(f"  Reviewer:     {stats['reviewer']}")
    print(f"  Presented:    {stats['total_presented']}")
    print(f"  Labeled:      {total_labeled}")
    print(f"    AI:         {stats['labeled_ai']}")
    print(f"    Human:      {stats['labeled_human']}")
    print(f"    Unsure:     {stats['labeled_unsure']}")
    print(f"  Skipped:      {stats['skipped']}")

    tp = stats['true_positives']
    fp = stats['false_positives']
    tn = stats['true_negatives']
    fn = stats['false_negatives']

    if tp + fp + tn + fn > 0:
        print(f"\n  --- Pipeline Accuracy (this session) ---")
        print(f"  True Positives:  {tp}  (flagged + confirmed AI)")
        print(f"  False Positives: {fp}  (flagged + confirmed human)")
        print(f"  True Negatives:  {tn}  (clean + confirmed human)")
        print(f"  False Negatives: {fn}  (clean + confirmed AI)")

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

        print(f"\n  Precision:  {precision:.1%}")
        print(f"  Recall:     {recall:.1%}")
        print(f"  Accuracy:   {accuracy:.1%}")

        if fp > 0:
            fpr = fp / max(fp + tn, 1)
            print(f"  FPR:        {fpr:.1%}")

    # Calibration readiness
    human_count = stats['labeled_human']
    print(f"\n  --- Calibration Status ---")
    if human_count >= 20:
        print(f"  READY: {human_count} human labels (minimum 20 met)")
        print(f"  Run: --calibrate {output_path}")
    elif human_count >= 10:
        print(f"  CLOSE: {human_count}/20 human labels — need {20 - human_count} more")
    else:
        print(f"  BUILDING: {human_count}/20 human labels — need {20 - human_count} more")

    print(f"\n  Labels saved to: {output_path}")


# ==============================================================================
# CALIBRATION REPORT
# ==============================================================================


def calibration_report(jsonl_path, cal_table=None, output_csv=None):
    """Generate a calibration diagnostics report from labeled data."""
    import statistics

    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if rec.get('ground_truth') in ('ai', 'human'):
                        records.append(rec)
                except json.JSONDecodeError:
                    continue

    if len(records) < 5:
        print(f"  Insufficient labeled data ({len(records)} records, need >= 5)")
        return None

    n_total = len(records)
    n_human = sum(1 for r in records if r['ground_truth'] == 'human')
    n_ai = sum(1 for r in records if r['ground_truth'] == 'ai')

    print(f"\n{'='*80}")
    print(f"  CALIBRATION DIAGNOSTICS REPORT")
    print(f"  {n_total} labeled records ({n_human} human, {n_ai} AI)")
    print(f"{'='*80}")

    # 1. Confusion matrix by determination
    print(f"\n  --- Confusion Matrix by Determination ---")
    print(f"  {'Determination':<12} {'Human':>8} {'AI':>8} {'Total':>8} {'Precision':>10}")
    print(f"  {'-'*50}")

    for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'GREEN', 'REVIEW']:
        det_records = [r for r in records if r.get('pipeline_determination') == det]
        if not det_records:
            continue
        det_human = sum(1 for r in det_records if r['ground_truth'] == 'human')
        det_ai = sum(1 for r in det_records if r['ground_truth'] == 'ai')
        det_total = len(det_records)
        precision = det_ai / det_total if det in ('RED', 'AMBER', 'MIXED') else det_human / det_total
        print(f"  {det:<12} {det_human:>8} {det_ai:>8} {det_total:>8} {precision:>9.1%}")

    # 2. Reliability diagram (confidence bins)
    print(f"\n  --- Reliability Diagram (Confidence Bins) ---")
    print(f"  {'Bin':<12} {'Count':>6} {'AI Rate':>8} {'Mean Conf':>10} {'Gap':>8}")
    print(f"  {'-'*50}")

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    reliability_gaps = []

    for lo, hi in bins:
        bin_records = [r for r in records
                        if lo <= r.get('pipeline_confidence', r.get('confidence', 0)) < hi]
        if not bin_records:
            continue

        n_bin = len(bin_records)
        ai_rate = sum(1 for r in bin_records if r['ground_truth'] == 'ai') / n_bin
        mean_conf = statistics.mean(
            r.get('pipeline_confidence', r.get('confidence', 0)) for r in bin_records)
        gap = mean_conf - ai_rate
        reliability_gaps.append(abs(gap))

        print(f"  [{lo:.1f}-{hi:.1f})  {n_bin:>6} {ai_rate:>7.1%} {mean_conf:>9.2f} "
              f"{'':>2}{'+' if gap > 0 else ''}{gap:>5.2f}")

    ece = None
    if reliability_gaps:
        ece = statistics.mean(reliability_gaps)
        print(f"\n  Expected Calibration Error (ECE): {ece:.3f}")
        if ece < 0.10:
            print(f"  Assessment: WELL CALIBRATED")
        elif ece < 0.20:
            print(f"  Assessment: MODERATELY CALIBRATED — consider recalibration")
        else:
            print(f"  Assessment: POORLY CALIBRATED — recalibration recommended")

    # 3. TPR at fixed FPR
    if n_human >= 5 and n_ai >= 5:
        print(f"\n  --- TPR @ Fixed FPR ---")

        scores = [r.get('pipeline_confidence', r.get('confidence', 0)) for r in records]
        labels = [1 if r['ground_truth'] == 'ai' else 0 for r in records]

        thresholds = sorted(set(scores), reverse=True)

        for target_fpr, label in [(0.01, '1%'), (0.05, '5%'), (0.10, '10%')]:
            best_tpr = 0.0
            best_thresh = 1.0
            for t in thresholds:
                predicted_pos = [s >= t for s in scores]
                fp = sum(1 for p, l in zip(predicted_pos, labels) if p and l == 0)
                tp = sum(1 for p, l in zip(predicted_pos, labels) if p and l == 1)
                fpr = fp / max(n_human, 1)
                tpr = tp / max(n_ai, 1)
                if fpr <= target_fpr and tpr > best_tpr:
                    best_tpr = tpr
                    best_thresh = t
            print(f"  TPR @ {label:>3} FPR: {best_tpr:>6.1%}  (threshold={best_thresh:.3f})")

    # 4. Per-stratum analysis
    strata = defaultdict(list)
    for r in records:
        domain = r.get('domain', 'unknown') or 'unknown'
        length_bin = r.get('length_bin', 'unknown') or 'unknown'
        strata[f"{domain}x{length_bin}"].append(r)

    if len(strata) > 1:
        print(f"\n  --- Per-Stratum Calibration ---")
        print(f"  {'Stratum':<30} {'N':>5} {'FP Rate':>8} {'FN Rate':>8} {'Flag Rate':>10}")
        print(f"  {'-'*65}")

        problem_strata = []

        for stratum_key in sorted(strata.keys()):
            s_records = strata[stratum_key]
            if len(s_records) < 3:
                continue

            s_human = [r for r in s_records if r['ground_truth'] == 'human']
            s_ai = [r for r in s_records if r['ground_truth'] == 'ai']

            fp = sum(1 for r in s_human
                      if r.get('pipeline_determination') in ('RED', 'AMBER', 'MIXED'))
            fn = sum(1 for r in s_ai
                      if r.get('pipeline_determination') in ('GREEN', 'YELLOW'))

            fpr = fp / max(len(s_human), 1)
            fnr = fn / max(len(s_ai), 1)

            flagged = sum(1 for r in s_records
                           if r.get('pipeline_determination') in ('RED', 'AMBER', 'MIXED'))
            flag_rate = flagged / len(s_records)

            print(f"  {stratum_key:<30} {len(s_records):>5} {fpr:>7.1%} {fnr:>7.1%} {flag_rate:>9.1%}")

            if fpr > 0.15 or fnr > 0.30:
                problem_strata.append((stratum_key, fpr, fnr))

        if problem_strata:
            print(f"\n  WARNING: Problem strata detected:")
            for sk, fpr, fnr in problem_strata:
                issues = []
                if fpr > 0.15:
                    issues.append(f"high FPR ({fpr:.0%})")
                if fnr > 0.30:
                    issues.append(f"high FNR ({fnr:.0%})")
                print(f"    {sk}: {', '.join(issues)}")

    # 5. False positive analysis
    false_positives = [r for r in records
                        if r['ground_truth'] == 'human'
                        and r.get('pipeline_determination') in ('RED', 'AMBER', 'MIXED')]

    if false_positives:
        print(f"\n  --- False Positive Analysis ({len(false_positives)} cases) ---")

        fp_confs = [r.get('pipeline_confidence', r.get('confidence', 0))
                     for r in false_positives]
        print(f"  Confidence range: {min(fp_confs):.2f} - {max(fp_confs):.2f} "
              f"(mean={statistics.mean(fp_confs):.2f})")

        fp_dets = Counter(r.get('pipeline_determination') for r in false_positives)
        for det, count in fp_dets.most_common():
            print(f"    {det}: {count}")

        fp_modes = Counter(r.get('mode', '?') for r in false_positives)
        if len(fp_modes) > 1:
            print(f"  By mode: {dict(fp_modes)}")

    # 6. Recommendations
    print(f"\n  --- Recommendations ---")

    if n_human < 20:
        print(f"  [!] Need {20 - n_human} more human labels for conformal calibration")
    else:
        print(f"  [OK] Sufficient human labels ({n_human}) for calibration")
        print(f"       Run: --calibrate {jsonl_path}")

    if false_positives and len(false_positives) / max(n_human, 1) > 0.10:
        print(f"  [!] FPR > 10% — consider raising detection thresholds")

    false_negatives = [r for r in records
                        if r['ground_truth'] == 'ai'
                        and r.get('pipeline_determination') in ('GREEN', 'YELLOW')]
    if false_negatives and len(false_negatives) / max(n_ai, 1) > 0.20:
        print(f"  [!] FNR > 20% — review missed cases for new signal patterns")

    # Output CSV if requested
    if output_csv:
        pd.DataFrame(records).to_csv(output_csv, index=False)
        print(f"\n  Labeled data exported to: {output_csv}")

    return {
        'n_total': n_total,
        'n_human': n_human,
        'n_ai': n_ai,
        'ece': ece,
        'false_positives': len(false_positives) if false_positives else 0,
        'false_negatives': len(false_negatives) if false_negatives else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='LLM Detection Pipeline v0.66')
    parser.add_argument('input', nargs='?', help='Input file (.xlsx, .csv, or .pdf)')
    parser.add_argument('--gui', action='store_true', help='Launch desktop GUI mode')
    parser.add_argument('--web', action='store_true', help='Launch Streamlit web dashboard')
    parser.add_argument('--text', help='Analyze a single text string')
    parser.add_argument('--sheet', help='Sheet name for xlsx files')
    parser.add_argument('--prompt-col', default='prompt', help='Column name containing prompts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all layer details')
    parser.add_argument('--output', '-o', help='Output CSV path')
    parser.add_argument('--attempter', help='Filter by attempter name (substring match)')
    parser.add_argument('--no-similarity', action='store_true',
                        help='Skip cross-submission similarity analysis')
    parser.add_argument('--similarity-threshold', type=float, default=0.40,
                        help='Jaccard threshold for text similarity (default: 0.40)')
    parser.add_argument('--collect', metavar='PATH',
                        help='Append scored results to JSONL file for baseline accumulation')
    parser.add_argument('--analyze-baselines', metavar='JSONL',
                        help='Read accumulated JSONL and print per-occupation percentile tables')
    parser.add_argument('--baselines-csv', metavar='PATH',
                        help='Write baseline percentile tables to CSV (use with --analyze-baselines)')
    parser.add_argument('--no-layer3', action='store_true',
                        help='Skip API continuation analysis entirely (NSSI + DNA-GPT)')
    parser.add_argument('--disable-channel', metavar='CHANNELS',
                        help='Comma-separated channel names to disable for ablation: '
                             'prompt_structure, stylometry, continuation, windowing')
    parser.add_argument('--api-key', metavar='KEY',
                        help='API key for DNA-GPT continuation analysis. Falls back to '
                             'ANTHROPIC_API_KEY or OPENAI_API_KEY env var.')
    parser.add_argument('--provider', default='anthropic', choices=['anthropic', 'openai'],
                        help='LLM provider for DNA-GPT (default: anthropic)')
    parser.add_argument('--dna-model', metavar='MODEL',
                        help='Model name for DNA-GPT (default: auto per provider)')
    parser.add_argument('--ppl-model', metavar='MODEL',
                        help='HuggingFace model for perplexity scoring '
                             '(default: Qwen/Qwen2.5-0.5B). '
                             'Options: Qwen/Qwen2.5-0.5B, HuggingFaceTB/SmolLM2-360M, '
                             'HuggingFaceTB/SmolLM2-135M, distilgpt2, gpt2')
    parser.add_argument('--dna-samples', type=int, default=3,
                        help='Number of regeneration samples for DNA-GPT (default: 3)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for batch processing (default: 1)')
    parser.add_argument('--batch', action='store_true',
                        help='Use Anthropic Message Batches API for continuation analysis '
                             '(50%% cheaper, processes all submissions in one server-side batch)')
    parser.add_argument('--mode', default='auto', choices=['task_prompt', 'generic_aigt', 'auto'],
                        help='Detection mode: task_prompt (prompt-structure primary), '
                             'generic_aigt (all channels), auto (heuristic). Default: auto')
    parser.add_argument('--calibrate', metavar='JSONL',
                        help='Build calibration table from labeled baseline JSONL and save to --cal-table')
    parser.add_argument('--cal-table', metavar='JSON',
                        help='Path to calibration table JSON (load for scoring, or save target for --calibrate)')
    parser.add_argument('--cost-per-prompt', type=float, default=400.0,
                        help='Cost per prompt for financial impact estimate (default: $400)')
    parser.add_argument('--html-report', metavar='FILE',
                        help='Generate a consolidated HTML report for flagged submissions')
    parser.add_argument('--similarity-store', metavar='JSONL',
                        help='Path to persistent similarity fingerprint store (cross-batch)')
    parser.add_argument('--instructions', metavar='FILE',
                        help='Path to shared project instructions file (for similarity baseline)')
    parser.add_argument('--memory', metavar='DIR', default=None,
                        help='Path to BEET memory store directory (enables cross-batch memory)')
    parser.add_argument('--confirm', nargs=3, metavar=('TASK_ID', 'LABEL', 'REVIEWER'),
                        help='Record a ground truth confirmation: --confirm task_001 ai reviewer_A')
    parser.add_argument('--attempter-history', metavar='NAME',
                        help='Show historical profile for an attempter')
    parser.add_argument('--memory-summary', action='store_true',
                        help='Print memory store summary')
    parser.add_argument('--rebuild-calibration', action='store_true',
                        help='Rebuild calibration table from confirmed labels in memory')
    parser.add_argument('--rebuild-shadow', action='store_true',
                        help='Rebuild shadow model from confirmed labels in memory')
    parser.add_argument('--discover-lexicon', action='store_true',
                        help='Run log-odds lexicon discovery on confirmed labels')
    parser.add_argument('--rebuild-centroids', action='store_true',
                        help='Rebuild semantic centroids from confirmed labels')
    parser.add_argument('--rebuild-all', action='store_true',
                        help='Rebuild calibration, shadow model, and centroids')
    parser.add_argument('--labeled-corpus', metavar='JSONL',
                        help='Path to JSONL with raw text for lexicon discovery and centroids')
    parser.add_argument('--label', action='store_true',
                        help='Interactive labeling mode: review results and assign '
                             'ground truth labels for calibration')
    parser.add_argument('--label-output', metavar='JSONL',
                        help='JSONL path for labeled records (default: auto-named)')
    parser.add_argument('--label-reviewer', metavar='NAME',
                        help='Reviewer name/ID for labeling session')
    parser.add_argument('--label-skip-green', action='store_true',
                        help='Skip GREEN determinations during labeling (assume correct)')
    parser.add_argument('--label-skip-red', action='store_true',
                        help='Skip RED determinations during labeling (assume correct)')
    parser.add_argument('--label-max', type=int, metavar='N',
                        help='Maximum number of items to label per session')
    parser.add_argument('--calibration-report', metavar='JSONL',
                        help='Generate calibration diagnostics report from labeled JSONL')
    parser.add_argument('--calibration-report-csv', metavar='PATH',
                        help='Export labeled data to CSV (use with --calibration-report)')
    # Column mapping
    parser.add_argument('--id-col', default='task_id', metavar='COL',
                        help='Column name or letter (A–Z) for task ID (default: task_id)')
    parser.add_argument('--occ-col', default='occupation', metavar='COL',
                        help='Column name or letter (A–Z) for occupation/area (default: occupation)')
    parser.add_argument('--attempter-col', default='attempter_name', metavar='COL',
                        help='Column name or letter (A–Z) for attempter/author (default: attempter_name)')
    parser.add_argument('--stage-col', default='pipeline_stage_name', metavar='COL',
                        help='Column name or letter (A–Z) for pipeline stage (default: pipeline_stage_name)')
    parser.add_argument('--attempter-email-col', default='', metavar='COL',
                        help='Column name or letter (A–Z) for attempter email (optional)')
    parser.add_argument('--reviewer-col', default='', metavar='COL',
                        help='Column name or letter (A–Z) for reviewer name (optional)')
    parser.add_argument('--reviewer-email-col', default='', metavar='COL',
                        help='Column name or letter (A–Z) for reviewer email (optional)')
    # Output directory
    parser.add_argument('--run-dir', metavar='DIR',
                        help='Root directory for this analysis run. A timestamped subfolder '
                             '(run_YYYYMMDD_HHMMSS) is created automatically and all outputs '
                             '(results CSV, HTML report, memory store, similarity store, '
                             'labels) are saved there. Individual path flags override the '
                             'auto-generated paths.')
    args = parser.parse_args()

    if args.gui:
        from llm_detector.gui import launch_gui
        launch_gui()
        return

    if args.web:
        main_dashboard()
        return

    # ── Run-directory: create timestamped folder and set default output paths ─
    if args.run_dir:
        run_dir = Path(args.run_dir) / datetime.now().strftime('run_%Y%m%d_%H%M%S')
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Run directory: {run_dir}")
        if not args.output:
            args.output = str(run_dir / 'results.csv')
        if not args.html_report:
            args.html_report = str(run_dir / 'report.html')
        if not args.memory:
            args.memory = str(run_dir / 'memory')
        if not args.similarity_store:
            args.similarity_store = str(run_dir / 'similarity.jsonl')
        if not getattr(args, 'label_output', None):
            args.label_output = str(run_dir / 'labels.jsonl')

    # Memory store setup
    store = None
    if args.memory:
        from llm_detector.memory import MemoryStore
        store = MemoryStore(args.memory)

    # Memory-only commands (early exit)
    if args.memory_summary:
        if store:
            store.print_summary()
        else:
            print("ERROR: --memory-summary requires --memory DIR")
        return

    if args.confirm:
        if store:
            task_id, label, reviewer = args.confirm
            if label not in ('ai', 'human'):
                print(f"ERROR: label must be 'ai' or 'human', got '{label}'")
                return
            store.record_confirmation(task_id, label, verified_by=reviewer)
        else:
            print("ERROR: --confirm requires --memory DIR")
        return

    if args.attempter_history:
        if store:
            store.print_attempter_history(args.attempter_history)
        else:
            print("ERROR: --attempter-history requires --memory DIR")
        return

    if args.rebuild_calibration:
        if store:
            cal = store.rebuild_calibration()
            if cal:
                print(f"  Calibration rebuilt: {cal['n_calibration']} labeled samples")
        else:
            print("ERROR: --rebuild-calibration requires --memory DIR")
        return

    if args.rebuild_shadow:
        if store:
            shadow = store.rebuild_shadow_model()
            if shadow:
                print(f"  Shadow model built: AUC={shadow['cv_auc']:.3f}")
        else:
            print("ERROR: --rebuild-shadow requires --memory DIR")
        return

    if args.discover_lexicon:
        if not store:
            print("ERROR: --discover-lexicon requires --memory DIR")
            return
        if not args.labeled_corpus:
            print("ERROR: --discover-lexicon requires --labeled-corpus")
            return
        store.discover_lexicon_candidates(args.labeled_corpus)
        return

    if args.rebuild_centroids:
        if not store:
            print("ERROR: --rebuild-centroids requires --memory DIR")
            return
        if not args.labeled_corpus:
            print("ERROR: --rebuild-centroids requires --labeled-corpus")
            return
        result = store.rebuild_semantic_centroids(args.labeled_corpus)
        if result:
            print(f"  Centroid separation: {result['separation']:.4f}")
        return

    if args.rebuild_all:
        if not store:
            print("ERROR: --rebuild-all requires --memory DIR")
            return

        print(f"\n{'='*70}")
        print(f"  REBUILDING ALL LEARNED ARTIFACTS")
        print(f"{'='*70}")

        cal = store.rebuild_calibration()
        if cal:
            print(f"  + Calibration: {cal['n_calibration']} samples")
        else:
            print(f"  - Calibration: insufficient data")

        shadow = store.rebuild_shadow_model()
        if shadow:
            print(f"  + Shadow model: AUC={shadow['cv_auc']:.3f}")
        else:
            print(f"  - Shadow model: insufficient labeled data")

        if args.labeled_corpus:
            centroids = store.rebuild_semantic_centroids(args.labeled_corpus)
            if centroids:
                print(f"  + Centroids: separation={centroids['separation']:.4f}")
            else:
                print(f"  - Centroids: insufficient labeled text")
        else:
            print(f"  - Centroids: skipped (no --labeled-corpus)")

        if args.labeled_corpus:
            candidates = store.discover_lexicon_candidates(args.labeled_corpus)
            n_new = sum(1 for c in candidates
                        if not c.get('already_in_fingerprints')
                        and not c.get('already_in_packs'))
            print(f"  + Lexicon: {len(candidates)} candidates ({n_new} new)")
        else:
            print(f"  - Lexicon: skipped (no --labeled-corpus)")

        print(f"\n{'='*70}")
        return

    if not args.api_key:
        env_key = 'ANTHROPIC_API_KEY' if args.provider == 'anthropic' else 'OPENAI_API_KEY'
        args.api_key = os.environ.get(env_key)

    if args.analyze_baselines:
        if not os.path.exists(args.analyze_baselines):
            print(f"ERROR: File not found: {args.analyze_baselines}")
            return
        analyze_baselines(args.analyze_baselines, output_csv=args.baselines_csv)
        return

    if args.calibrate:
        if not os.path.exists(args.calibrate):
            print(f"ERROR: File not found: {args.calibrate}")
            return
        cal = calibrate_from_baselines(args.calibrate)
        if cal is None:
            print("ERROR: Insufficient labeled human data for calibration (need >=20)")
            return
        cal_path = args.cal_table or args.calibrate.replace('.jsonl', '_calibration.json')
        save_calibration(cal, cal_path)
        print(f"  Global quantiles: {cal['global']}")
        print(f"  Strata: {len(cal.get('strata', {}))} domain x length_bin tables")
        return

    cal_table = None
    if args.cal_table and os.path.exists(args.cal_table):
        cal_table = load_calibration(args.cal_table)
        print(f"Loaded calibration table: {cal_table['n_calibration']} records, "
              f"{len(cal_table.get('strata', {}))} strata")

    if args.calibration_report:
        if not os.path.exists(args.calibration_report):
            print(f"ERROR: File not found: {args.calibration_report}")
            return
        calibration_report(
            args.calibration_report,
            cal_table=cal_table,
            output_csv=getattr(args, 'calibration_report_csv', None),
        )
        return

    run_l3 = not args.no_layer3

    disabled_channels = set()
    if args.disable_channel:
        disabled_channels = {c.strip() for c in args.disable_channel.split(',')}
        valid = {'prompt_structure', 'stylometry', 'continuation', 'windowing'}
        invalid = disabled_channels - valid
        if invalid:
            print(f"WARNING: Unknown channel names: {invalid}. Valid: {valid}")
            disabled_channels &= valid
        if disabled_channels:
            print(f"  Channels disabled (ablation): {', '.join(sorted(disabled_channels))}")

    if args.text:
        result = analyze_prompt(
            args.text, run_l3=run_l3,
            api_key=args.api_key, dna_provider=args.provider,
            dna_model=args.dna_model, dna_samples=args.dna_samples,
            mode=args.mode, cal_table=cal_table,
            disabled_channels=disabled_channels,
            ppl_model=getattr(args, 'ppl_model', None),
        )
        print_result(result, verbose=True)
        return

    if not args.input:
        if _is_frozen():
            from llm_detector.gui import launch_gui
            launch_gui()
            return
        parser.print_help()
        return

    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.xlsx', '.xlsm'):
        tasks = load_xlsx(
            args.input, sheet=args.sheet,
            prompt_col=args.prompt_col,
            id_col=args.id_col,
            occ_col=args.occ_col,
            attempter_col=args.attempter_col,
            stage_col=args.stage_col,
            attempter_email_col=args.attempter_email_col,
            reviewer_col=args.reviewer_col,
            reviewer_email_col=args.reviewer_email_col,
        )
    elif ext == '.csv':
        tasks = load_csv(
            args.input,
            prompt_col=args.prompt_col,
            id_col=args.id_col,
            occ_col=args.occ_col,
            attempter_col=args.attempter_col,
            stage_col=args.stage_col,
            attempter_email_col=args.attempter_email_col,
            reviewer_col=args.reviewer_col,
            reviewer_email_col=args.reviewer_email_col,
        )
    elif ext == '.pdf':
        tasks = load_pdf(args.input)
    else:
        print(f"ERROR: Unsupported file type: {ext}")
        return

    if not tasks:
        print("ERROR: No tasks found.")
        return

    if args.attempter:
        tasks = [t for t in tasks if args.attempter.lower() in t.get('attempter', '').lower()]
        print(f"Filtered to {len(tasks)} tasks matching attempter '{args.attempter}'")

    layer3_label = " + L3" if run_l3 else ""
    use_batch = getattr(args, 'batch', False) and args.api_key and args.provider == 'anthropic'
    dna_label = " + DNA-GPT (batch)" if use_batch else (" + DNA-GPT" if args.api_key else "")
    print(f"Processing {len(tasks)} tasks through pipeline v0.66{layer3_label}{dna_label}...")

    results = []
    text_map = {}

    # Build text_map upfront (needed regardless of parallelism)
    for i, task in enumerate(tasks):
        tid = task.get('task_id', f'_row{i}')
        text_map[tid] = task['prompt']

    # ── Batch API pre-computation (Anthropic only) ─────────────────
    batch_cont_results = {}
    if use_batch and run_l3:
        from llm_detector.analyzers.continuation_api import run_continuation_batch
        from llm_detector.normalize import normalize_text
        # Normalize texts the same way the pipeline does
        norm_texts = []
        norm_ids = []
        for task in tasks:
            nt, _ = normalize_text(task['prompt'])
            norm_texts.append(nt)
            norm_ids.append(task.get('task_id', ''))
        batch_cont_results = run_continuation_batch(
            norm_texts, norm_ids,
            api_key=args.api_key,
            model=args.dna_model,
            n_samples=args.dna_samples,
            progress_fn=lambda s: print(f"  {s}"),
        )
        print(f"  Batch complete: {len(batch_cont_results)} continuation results received.")

    n_workers = max(1, getattr(args, 'workers', 1))

    def _analyze_task(idx_task):
        i, task = idx_task
        precomputed = batch_cont_results.get(i)
        return i, analyze_prompt(
            task['prompt'],
            task_id=task.get('task_id', ''),
            occupation=task.get('occupation', ''),
            attempter=task.get('attempter', ''),
            stage=task.get('stage', ''),
            run_l3=run_l3,
            api_key=None if use_batch else args.api_key,
            dna_provider=args.provider,
            dna_model=args.dna_model,
            dna_samples=args.dna_samples,
            mode=args.mode,
            cal_table=cal_table,
            memory_store=store,
            disabled_channels=disabled_channels,
            precomputed_continuation=precomputed,
            ppl_model=getattr(args, 'ppl_model', None),
        )

    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        indexed_results = [None] * len(tasks)
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i, r in pool.map(_analyze_task, enumerate(tasks)):
                indexed_results[i] = r
                done += 1
                if done % 10 == 0:
                    print(f"  Processed {done}/{len(tasks)}...")
        results = indexed_results
    else:
        for i, task in enumerate(tasks):
            _, r = _analyze_task((i, task))
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(tasks)}...")

    det_counts = Counter(r['determination'] for r in results)
    print(f"\n{'='*90}")
    print(f"  PIPELINE v0.66 RESULTS (n={len(results)})")
    print(f"{'='*90}")
    all_dets = ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']
    icons = {
        'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1',
        'GREEN': '\U0001f7e2', 'MIXED': '\U0001f535', 'REVIEW': '\u26aa',
    }
    for det in all_dets:
        ct = det_counts.get(det, 0)
        if ct > 0 or det in ('RED', 'AMBER', 'YELLOW', 'GREEN'):
            pct = ct / len(results) * 100
            print(f"  {icons[det]} {det:>8}: {ct:>4} ({pct:.1f}%)")

    flagged = [r for r in results if r['determination'] in ('RED', 'AMBER', 'MIXED')]
    if flagged:
        print(f"\n{'='*90}")
        print(f"  FLAGGED SUBMISSIONS: {len(flagged)}")
        print(f"{'='*90}")
        for r in sorted(flagged, key=lambda x: x['confidence'], reverse=True):
            print_result(r, verbose=args.verbose)

    yellow = [r for r in results if r['determination'] == 'YELLOW']
    if yellow:
        print(f"\n  YELLOW ({len(yellow)} minor signals):")
        for r in sorted(yellow, key=lambda x: x['confidence'], reverse=True)[:10]:
            print(f"    \U0001f7e1 {r['task_id'][:12]:12} {r['occupation'][:40]:40} | {r['reason'][:50]}")

    # Load instruction text for similarity baseline (FEAT 15)
    instruction_text = None
    if args.instructions and os.path.exists(args.instructions):
        with open(args.instructions, 'r') as f:
            instruction_text = f.read()
        print(f"  Loaded instruction template ({len(instruction_text)} chars) for similarity baseline")

    if not args.no_similarity and len(results) >= 2:
        sim_pairs = analyze_similarity(
            results, text_map,
            jaccard_threshold=args.similarity_threshold,
            instruction_text=instruction_text,
        )
        print_similarity_report(sim_pairs)

        # FEAT 13: Similarity feedback into determination
        if sim_pairs:
            results = apply_similarity_adjustments(results, sim_pairs, text_map)
            upgrades = [r for r in results if 'similarity_upgrade' in r]
            if upgrades:
                det_counts = Counter(r['determination'] for r in results)
                print(f"\n  SIMILARITY ADJUSTMENTS: {len(upgrades)} determinations upgraded")
                for r in upgrades:
                    su = r['similarity_upgrade']
                    print(f"    {r['task_id'][:15]:15s} {su['original_determination']} -> "
                          f"{su['upgraded_to']}  ({su['reason'][:60]})")
    else:
        sim_pairs = []

    # FEAT 14: Cross-batch similarity store
    if args.similarity_store:
        cross_flags = cross_batch_similarity(
            results, text_map, args.similarity_store
        )
        if cross_flags:
            print(f"\n  CROSS-BATCH SIMILARITY: {len(cross_flags)} matches to previous batches")
            for cf in cross_flags[:10]:
                print(f"    {cf['current_id'][:15]} <-> {cf['historical_id'][:15]} "
                      f"(MH={cf['minhash_similarity']:.2f}, batch={cf['historical_batch'][:10]})")
        save_similarity_store(results, text_map, args.similarity_store)

    # Shadow model disagreement check (if memory store has a trained model)
    if store:
        shadow_count = 0
        for r in results:
            disagreement = store.check_shadow_disagreement(r)
            r['shadow_disagreement'] = disagreement
            r['shadow_ai_prob'] = (disagreement or {}).get('shadow_ai_prob')
            if disagreement:
                shadow_count += 1
        if shadow_count:
            print(f"\n  SHADOW MODEL: {shadow_count} disagreements with rule engine")

    # Memory store: cross-batch similarity + record batch
    if store:
        cross_flags = store.cross_batch_similarity(results, text_map)
        if cross_flags:
            print(f"\n  CROSS-BATCH MEMORY: {len(cross_flags)} matches to previous submissions")
            for cf in cross_flags[:5]:
                print(f"    {cf['current_id'][:15]} <-> {cf['historical_id'][:15]} "
                      f"(MH={cf['minhash_similarity']:.2f}, batch={cf['historical_batch'][:15]})")
        store.record_batch(results, text_map)

    if getattr(args, 'label', False):
        label_stats = interactive_label(
            results, text_map,
            output_path=getattr(args, 'label_output', None),
            reviewer=getattr(args, 'label_reviewer', '') or '',
            store=store,
            skip_green=getattr(args, 'label_skip_green', False),
            skip_red=getattr(args, 'label_skip_red', False),
            max_labels=getattr(args, 'label_max', None),
        )

        # Auto-calibrate if enough labels collected
        label_path = getattr(args, 'label_output', None) or \
                     f"beet_labels_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        if os.path.exists(label_path):
            human_count = label_stats.get('labeled_human', 0)
            if human_count >= 20:
                print(f"\n  Sufficient labels for calibration. Building table...")
                cal = calibrate_from_baselines(label_path)
                if cal:
                    cal_out = label_path.replace('.jsonl', '_calibration.json')
                    save_calibration(cal, cal_out)

    default_name = os.path.basename(args.input).rsplit('.', 1)[0] + '_pipeline_v066.csv'
    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_path = args.output or os.path.join(input_dir, default_name)

    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'preamble_details'}
        row['preamble_details'] = str(r.get('preamble_details', []))
        flat.append(row)

    if sim_pairs:
        sim_lookup = defaultdict(list)
        for p in sim_pairs:
            # Skip baseline statistics that don't have id_a/id_b
            if p.get('_type') == 'baseline':
                continue
            sim_lookup[p['id_a']].append(f"{p['id_b']}(J={p['jaccard']:.2f})")
            sim_lookup[p['id_b']].append(f"{p['id_a']}(J={p['jaccard']:.2f})")
        for row in flat:
            tid = row.get('task_id', '')
            row['similarity_flags'] = '; '.join(sim_lookup.get(tid, []))

    pd.DataFrame(flat).to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    # Attempter profiling and channel pattern summary
    if len(results) >= 5:
        from llm_detector.reporting import (
            profile_attempters, print_attempter_report, channel_pattern_summary,
        )
        profiles = profile_attempters(results)
        print_attempter_report(profiles)
        channel_pattern_summary(results)

    # Financial impact estimate
    if len(results) >= 10:
        from llm_detector.reporting import financial_impact, print_financial_report
        impact = financial_impact(results, cost_per_prompt=args.cost_per_prompt)
        print_financial_report(impact, cost_per_prompt=args.cost_per_prompt)

    # Consolidated HTML report for flagged submissions
    if args.html_report and flagged:
        report_path = args.html_report
        if not report_path.endswith('.html'):
            report_path += '.html'
        os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
        from llm_detector.html_report import generate_batch_html_report
        generate_batch_html_report(flagged, text_map, report_path)
        print(f"\n  HTML report written to {report_path} ({len(flagged)} submissions)")

    if args.collect:
        collect_baselines(results, args.collect)


def main_gui():
    """Entry point that always launches the GUI (for gui-scripts / executable)."""
    from llm_detector.gui import launch_gui
    launch_gui()


def _ensure_streamlit():
    """Auto-install streamlit if missing, especially for frozen/executable builds."""
    try:
        st_spec = importlib.util.find_spec('streamlit')
        if st_spec is not None:
            return True
    except (ImportError, ModuleNotFoundError):
        pass
    print('  Streamlit is not installed — installing automatically…')
    try:
        subprocess.check_call(
            [_real_python(), '-m', 'pip', 'install', _STREAMLIT_MIN_VERSION],
            stdout=subprocess.DEVNULL,
        )
        print('  ✅ Streamlit installed successfully.')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f'  ❌ Auto-install failed: {exc}')
        print('  Install manually with: pip install "llm-detector[web]"')
        return False


def main_dashboard():
    """Entry point that launches the Streamlit web dashboard."""
    spec = importlib.util.find_spec('llm_detector.dashboard')
    if spec is None or spec.origin is None:
        print('ERROR: llm_detector.dashboard module not found.')
        print('Ensure the llm_detector package is properly installed.')
        return
    dashboard_path = os.path.realpath(spec.origin)
    # Validate the resolved path points to a file within the llm_detector package
    pkg_dir = os.path.realpath(os.path.dirname(__file__))
    if not dashboard_path.startswith(pkg_dir + os.sep):
        print('ERROR: dashboard path is outside the llm_detector package.')
        return
    streamlit_exe = shutil.which('streamlit')
    if streamlit_exe:
        cmd = [streamlit_exe, 'run', dashboard_path]
    else:
        if not _ensure_streamlit():
            return
        # Re-check after install
        streamlit_exe = shutil.which('streamlit')
        if streamlit_exe:
            cmd = [streamlit_exe, 'run', dashboard_path]
        else:
            cmd = [_real_python(), '-m', 'streamlit', 'run', dashboard_path]
    subprocess.run(cmd, check=False)


def main_web():
    """Entry point for the ``llm-detector-dashboard`` console script.

    Launches the Streamlit web dashboard.  Equivalent to running::

        streamlit run llm_detector/dashboard.py
    """
    main_dashboard()


if __name__ == '__main__':
    main()
