# Bug Fix Guide - BUG-001

## Issue: Similarity Analysis KeyError

**File:** `llm_detector/cli.py`
**Line:** 1185
**Severity:** Medium
**Status:** ✅ FIXED (commit 2de79ab)

## Problem

When running batch CSV/XLSX processing with similarity analysis enabled (default), the CLI crashes with:

```
KeyError: 'id_a'
```

## Root Cause

The `analyze_similarity()` function returns a mixed list:
1. Pair dictionaries (with 'id_a', 'id_b' keys)
2. Baseline statistics (with '_type': 'baseline', NO 'id_a' or 'id_b')

The CLI iterates over ALL items without filtering baselines.

## Current Code (cli.py:1182-1186)

```python
if sim_pairs:
    sim_lookup = defaultdict(list)
    for p in sim_pairs:
        sim_lookup[p['id_a']].append(f"{p['id_b']}(J={p['jaccard']:.2f})")
        sim_lookup[p['id_b']].append(f"{p['id_a']}(J={p['jaccard']:.2f})")
```

## Recommended Fix

```python
if sim_pairs:
    sim_lookup = defaultdict(list)
    for p in sim_pairs:
        # Skip baseline statistics that don't have id_a/id_b
        if p.get('_type') == 'baseline':
            continue
        sim_lookup[p['id_a']].append(f"{p['id_b']}(J={p['jaccard']:.2f})")
        sim_lookup[p['id_b']].append(f"{p['id_a']}(J={p['jaccard']:.2f})")
```

## Workaround (Current)

Use the `--no-similarity` flag:

```bash
python -m llm_detector input.csv --no-similarity
```

## Test Case

After fix, add this integration test:

```python
def test_batch_csv_with_similarity():
    """Ensure similarity analysis doesn't crash on batch CSV input."""
    import tempfile
    import csv

    # Create test CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'prompt'])
        writer.writerow(['t1', 'Please create a detailed framework for evaluation'])
        writer.writerow(['t2', 'Develop a comprehensive authentication system'])
        writer.writerow(['t3', 'Write a test plan with explicit criteria'])
        csv_path = f.name

    # Should not crash
    from llm_detector.cli import main
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ['llm_detector', csv_path, '--prompt-col', 'prompt']
        main()
    finally:
        sys.argv = old_argv
        os.unlink(csv_path)
```

## Verification

After applying fix:

```bash
# Should work without crash
python -m llm_detector test.csv --prompt-col prompt

# Should still show similarity results
grep "similarity" output.csv
```

## Impact

- Fixes primary batch workflow
- Enables similarity analysis in batch mode
- No impact on single-text mode or --no-similarity mode
- Minimal code change (3 lines)

## Priority

**Medium** - Has workaround but blocks primary feature
**Effort:** 15 minutes
**Risk:** Low (simple guard clause)
