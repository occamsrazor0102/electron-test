"""Tests for reporting module (attempter profiling + financial impact)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.reporting import (
    profile_attempters, channel_pattern_summary, financial_impact,
)

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


def _make_result(attempter, determination, confidence=0.8, channels=None):
    """Helper to build a minimal pipeline result dict."""
    cd = {'channels': channels or {}}
    return {
        'attempter': attempter,
        'determination': determination,
        'confidence': confidence,
        'channel_details': cd,
        'task_id': f'{attempter}_task',
        'occupation': 'tester',
    }


def test_profile_empty():
    print("\n-- PROFILE: empty input --")
    profiles = profile_attempters([])
    check("empty results -> empty profiles", profiles == [])


def test_profile_ranks_by_flag_rate():
    print("\n-- PROFILE: ranks by flag_rate descending --")
    results = [
        _make_result('alice', 'RED'),
        _make_result('alice', 'RED'),
        _make_result('alice', 'GREEN'),
        _make_result('bob', 'RED'),
        _make_result('bob', 'RED'),
        _make_result('bob', 'RED'),
    ]
    profiles = profile_attempters(results, min_submissions=2)
    check("two profiles returned", len(profiles) == 2, f"got {len(profiles)}")
    check("bob first (100% flag rate)", profiles[0]['attempter'] == 'bob',
          f"got {profiles[0]['attempter']}")
    check("bob flag_rate == 1.0", profiles[0]['flag_rate'] == 1.0,
          f"got {profiles[0]['flag_rate']}")
    check("alice flag_rate ~0.667", abs(profiles[1]['flag_rate'] - 0.667) < 0.01,
          f"got {profiles[1]['flag_rate']}")


def test_profile_min_submissions():
    print("\n-- PROFILE: min_submissions filter --")
    results = [
        _make_result('alice', 'RED'),  # only 1 submission
        _make_result('bob', 'GREEN'),
        _make_result('bob', 'GREEN'),
    ]
    profiles = profile_attempters(results, min_submissions=2)
    check("alice excluded (< min_submissions)", len(profiles) == 1,
          f"got {len(profiles)}")
    if profiles:
        check("only bob remains", profiles[0]['attempter'] == 'bob')


def test_channel_pattern_summary():
    print("\n-- CHANNEL PATTERN SUMMARY --")
    results = [
        _make_result('a', 'RED', channels={
            'prompt_structure': {'severity': 'RED'},
            'stylometry': {'severity': 'GREEN'},
        }),
        _make_result('b', 'AMBER', channels={
            'prompt_structure': {'severity': 'AMBER'},
            'stylometry': {'severity': 'AMBER'},
        }),
        _make_result('c', 'GREEN'),  # should be skipped
    ]
    # channel_pattern_summary prints output; just verify it doesn't crash
    channel_pattern_summary(results)
    check("channel_pattern_summary ran without error", True)


def test_financial_impact_arithmetic():
    print("\n-- FINANCIAL IMPACT: arithmetic --")
    results = [
        _make_result('a', 'RED'),
        _make_result('b', 'RED'),
        _make_result('c', 'AMBER'),
        _make_result('d', 'GREEN'),
        _make_result('e', 'GREEN'),
    ]
    impact = financial_impact(results, cost_per_prompt=100.0)
    check("total_submissions == 5", impact['total_submissions'] == 5)
    check("total_spend == 500", impact['total_spend'] == 500.0,
          f"got {impact['total_spend']}")
    check("flagged_count == 3", impact['flagged_count'] == 3,
          f"got {impact['flagged_count']}")
    check("waste_estimate == 300", impact['waste_estimate'] == 300.0,
          f"got {impact['waste_estimate']}")
    check("flag_rate == 0.6", impact['flag_rate'] == 0.6,
          f"got {impact['flag_rate']}")
    check("clean_count == 2", impact['clean_count'] == 2)
    check("projected_annual_waste == 1200", impact['projected_annual_waste'] == 1200.0,
          f"got {impact['projected_annual_waste']}")


def test_financial_impact_empty():
    print("\n-- FINANCIAL IMPACT: empty (no divide-by-zero) --")
    impact = financial_impact([], cost_per_prompt=400.0)
    check("total_submissions == 0", impact['total_submissions'] == 0)
    check("flag_rate == 0.0", impact['flag_rate'] == 0.0)
    check("waste_estimate == 0.0", impact['waste_estimate'] == 0.0)
    check("no exception raised", True)


def test_profile_no_flagged_channels():
    """profile_attempters handles an attempter with no flagged channels."""
    print("\n-- PROFILE: no flagged channels --")
    results = [
        _make_result('alice', 'GREEN'),
        _make_result('alice', 'GREEN'),
    ]
    profiles = profile_attempters(results, min_submissions=1)
    check("one profile returned", len(profiles) == 1, f"got {len(profiles)}")
    check("primary_detection_channel is None",
          profiles[0].get('primary_detection_channel') is None,
          f"got {profiles[0].get('primary_detection_channel')}")

if __name__ == '__main__':
    print("=" * 70)
    print("  REPORTING MODULE TESTS")
    print("=" * 70)

    test_profile_empty()
    test_profile_ranks_by_flag_rate()
    test_profile_min_submissions()
    test_channel_pattern_summary()
    test_financial_impact_arithmetic()
    test_financial_impact_empty()
    test_profile_no_flagged_channels()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
