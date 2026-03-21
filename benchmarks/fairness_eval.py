"""Non-native English fairness evaluation.

Measures false positive rates across proficiency levels to validate
that the BEET language support gate properly limits over-flagging
of non-native English text.

Usage:
    python -m benchmarks.fairness_eval --corpus fairness_corpus.jsonl
    python -m benchmarks.fairness_eval --corpus fairness_corpus.jsonl --output results.csv

Expected JSONL format (one record per line):
    {"text": "...", "ground_truth": "human", "proficiency_level": "intermediate"}
    {"text": "...", "ground_truth": "human", "proficiency_level": "advanced"}
    {"text": "...", "ground_truth": "human", "proficiency_level": "native"}

proficiency_level values: beginner, intermediate, advanced, native
ground_truth values: human, ai
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict


def load_corpus(path):
    """Load JSONL corpus with text, ground_truth, and proficiency_level."""
    records = []
    with open(path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed line {i}")
                continue
            if 'text' not in rec:
                print(f"  WARNING: skipping line {i} (no 'text' field)")
                continue
            rec.setdefault('ground_truth', 'human')
            rec.setdefault('proficiency_level', 'unknown')
            records.append(rec)
    return records


def run_evaluation(corpus_path, output_csv=None):
    """Run BEET pipeline on corpus and report FPR by proficiency level."""
    from llm_detector.pipeline import analyze_prompt

    records = load_corpus(corpus_path)
    if not records:
        print(f"ERROR: no records found in {corpus_path}")
        return

    print(f"\n{'='*70}")
    print(f"  FAIRNESS EVALUATION -- {len(records)} texts from {corpus_path}")
    print(f"{'='*70}")

    # Tally by proficiency level
    level_counts = defaultdict(int)
    for r in records:
        level_counts[r['proficiency_level']] += 1
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count} texts")

    # Run pipeline
    results = []
    for i, rec in enumerate(records):
        result = analyze_prompt(
            rec['text'],
            task_id=f'fairness_{i}',
            ground_truth=rec['ground_truth'],
        )
        result['proficiency_level'] = rec['proficiency_level']
        result['input_ground_truth'] = rec['ground_truth']
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(records)}...")

    print(f"  Processed {len(results)}/{len(records)} total")

    # Compute FPR by proficiency level (for human-labeled texts only)
    print(f"\n{'='*70}")
    print(f"  FALSE POSITIVE RATES BY PROFICIENCY LEVEL")
    print(f"{'='*70}")
    print(f"  {'Level':<15} {'N':>5} {'Flagged':>8} {'FPR':>8} {'Severity Cap':>14}")
    print(f"  {'-'*52}")

    levels = sorted(set(r['proficiency_level'] for r in results))
    level_stats = {}

    for level in levels:
        level_results = [r for r in results
                         if r['proficiency_level'] == level
                         and r['input_ground_truth'] == 'human']
        if not level_results:
            continue

        n = len(level_results)
        flagged = sum(1 for r in level_results
                      if r['determination'] in ('RED', 'AMBER', 'MIXED'))
        fpr = flagged / n

        # Check severity cap activity
        capped = sum(1 for r in level_results
                     if r.get('lang_support_level') in ('REVIEW', 'UNSUPPORTED'))

        level_stats[level] = {
            'n': n, 'flagged': flagged, 'fpr': fpr, 'capped': capped,
        }
        cap_label = f"{capped}/{n} capped" if capped > 0 else "none"
        print(f"  {level:<15} {n:>5} {flagged:>8} {fpr:>7.1%} {cap_label:>14}")

    # Overall summary
    all_human = [r for r in results if r['input_ground_truth'] == 'human']
    if all_human:
        total_flagged = sum(1 for r in all_human
                            if r['determination'] in ('RED', 'AMBER', 'MIXED'))
        overall_fpr = total_flagged / len(all_human)
        print(f"\n  Overall FPR (human texts): {overall_fpr:.1%} ({total_flagged}/{len(all_human)})")

    # Disparity check
    if len(level_stats) >= 2:
        rates = [s['fpr'] for s in level_stats.values()]
        max_fpr = max(rates)
        min_fpr = min(rates)
        disparity = max_fpr - min_fpr
        if disparity > 0.10:
            worst = max(level_stats.items(), key=lambda x: x[1]['fpr'])
            print(f"\n  WARNING: FPR disparity of {disparity:.1%} across proficiency levels")
            print(f"  Highest FPR: {worst[0]} ({worst[1]['fpr']:.1%})")
        else:
            print(f"\n  FPR disparity: {disparity:.1%} (within 10pp tolerance)")

    # Optional CSV output
    if output_csv:
        try:
            import csv
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'task_id', 'proficiency_level', 'ground_truth', 'determination',
                    'confidence', 'lang_support_level', 'lang_fw_coverage',
                ])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        'task_id': r.get('task_id', ''),
                        'proficiency_level': r.get('proficiency_level', ''),
                        'ground_truth': r.get('input_ground_truth', ''),
                        'determination': r.get('determination', ''),
                        'confidence': r.get('confidence', 0),
                        'lang_support_level': r.get('lang_support_level', ''),
                        'lang_fw_coverage': r.get('lang_fw_coverage', 0),
                    })
            print(f"\n  Results written to {output_csv}")
        except Exception as e:
            print(f"\n  WARNING: could not write CSV: {e}")

    return level_stats


def main():
    parser = argparse.ArgumentParser(
        description='BEET Non-Native English Fairness Evaluation')
    parser.add_argument('--corpus', required=True,
                        help='Path to JSONL corpus with text, ground_truth, proficiency_level')
    parser.add_argument('--output', metavar='CSV',
                        help='Write per-text results to CSV')
    args = parser.parse_args()

    run_evaluation(args.corpus, output_csv=args.output)


if __name__ == '__main__':
    main()
