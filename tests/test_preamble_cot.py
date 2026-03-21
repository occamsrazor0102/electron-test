"""Tests for Chain-of-Thought leakage detection in preamble layer."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.analyzers.preamble import run_preamble

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


def test_think_tags():
    print("\n-- Think tag detection --")
    score, sev, hits, _spans = run_preamble(
        "<think>\nThe user wants a pharmacist task...\n</think>\n"
        "You are a board-certified pharmacist."
    )
    check("<think> tag -> CRITICAL", sev == 'CRITICAL', f"got {sev}")
    check("score == 0.99", score == 0.99, f"got {score}")
    hit_names = [h[0] for h in hits]
    check("cot_leakage in hits", 'cot_leakage' in hit_names, f"got {hit_names}")


def test_reasoning_tags():
    print("\n-- Reasoning tag detection --")
    score, sev, hits, _spans = run_preamble(
        "<reasoning>I need to design a complex scenario</reasoning>\n"
        "You are a senior data analyst."
    )
    check("<reasoning> tag -> CRITICAL", sev == 'CRITICAL', f"got {sev}")


def test_self_correction():
    print("\n-- Self-correction phrases --")
    score, sev, hits, _spans = run_preamble(
        "Wait, actually let me rethink the constraints for this task. "
        "You must process each CSV row and validate all fields."
    )
    check("Self-correction -> HIGH", sev == 'HIGH', f"got {sev}")
    check("score == 0.75", score == 0.75, f"got {score}")


def test_no_false_positive_think():
    print("\n-- No false positive on 'think carefully' --")
    score, sev, hits, _spans = run_preamble(
        "Think carefully about edge cases when reviewing patient records. "
        "You are a clinical pharmacist reviewing medication orders."
    )
    check("'Think carefully' -> no CoT hit", sev == 'NONE', f"got {sev}")


def test_no_false_positive_step():
    print("\n-- No false positive on legitimate numbered steps --")
    score, sev, hits, _spans = run_preamble(
        "Complete the following analysis in order:\n"
        "Step 1: Review the attached financial statements.\n"
        "Step 2: Identify discrepancies exceeding $10,000."
    )
    # step_numbering is MEDIUM, so this should fire but at low severity
    check("Legitimate steps -> MEDIUM at most", sev in ('NONE', 'MEDIUM'),
          f"got {sev}")


if __name__ == '__main__':
    print("=" * 70)
    print("  COT LEAKAGE PREAMBLE TESTS")
    print("=" * 70)

    test_think_tags()
    test_reasoning_tags()
    test_self_correction()
    test_no_false_positive_think()
    test_no_false_positive_step()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
