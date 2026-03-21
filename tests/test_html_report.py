"""Tests for HTML report generator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.html_report import generate_html_report, _apply_highlights

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


def test_apply_highlights_basic():
    print("\n-- APPLY HIGHLIGHTS: basic --")
    text = "You must include all fields."
    spans = [{'start': 4, 'end': 8, 'text': 'must', 'pack': 'obligation', 'weight': 1.0}]
    html_out = _apply_highlights(text, spans)
    check("contains signal class", 'class="signal' in html_out, f"got: {html_out[:100]}")
    check("contains 'must'", 'must' in html_out)
    check("contains title attr", 'title="obligation"' in html_out, f"got: {html_out[:200]}")


def test_apply_highlights_overlapping():
    print("\n-- APPLY HIGHLIGHTS: overlapping spans (highest severity wins) --")
    text = "MUST include"
    spans = [
        {'start': 0, 'end': 4, 'text': 'MUST', 'pack': 'obligation', 'weight': 1.0, 'type': 'pattern'},
        {'start': 0, 'end': 4, 'text': 'MUST', 'pack': 'obligation', 'weight': 2.0, 'type': 'uppercase'},
    ]
    html_out = _apply_highlights(text, spans)
    # uppercase has higher priority than pattern, so it should win
    check("uppercase class wins", 'signal-uppercase' in html_out, f"got: {html_out}")


def test_apply_highlights_empty():
    print("\n-- APPLY HIGHLIGHTS: empty spans --")
    text = "Hello <world> & 'friends'"
    html_out = _apply_highlights(text, [])
    check("no spans -> escaped text", '&lt;world&gt;' in html_out, f"got: {html_out}")
    check("ampersand escaped", '&amp;' in html_out)


def test_generate_html_report_returns_string():
    print("\n-- GENERATE HTML REPORT: returns string --")
    text = "You must include all required fields."
    result = {
        'detection_spans': [
            {'start': 4, 'end': 8, 'text': 'must', 'pack': 'obligation', 'weight': 1.0},
        ],
        'determination': 'RED',
        'reason': 'High obligation density',
        'confidence': 0.85,
        'task_id': 'test_001',
        'word_count': 7,
        'mode': 'task_prompt',
        'channel_details': {
            'channels': {
                'prompt_structure': {'severity': 'RED', 'explanation': 'High pack scores'},
                'stylometry': {'severity': 'GREEN', 'explanation': 'Normal'},
            },
        },
    }
    html_out = generate_html_report(text, result)
    check("returns string", isinstance(html_out, str))
    check("contains DOCTYPE", '<!DOCTYPE html>' in html_out)
    check("contains determination", 'RED' in html_out)
    check("contains task_id", 'test_001' in html_out)
    check("contains signal span", 'class="signal' in html_out)


def test_generate_html_report_channel_score_and_role():
    print("\n-- GENERATE HTML REPORT: score bars and role badges --")
    text = "You must complete this task."
    result = {
        'detection_spans': [],
        'determination': 'AMBER',
        'reason': 'Continuation RED single-channel demoted',
        'confidence': 0.70,
        'task_id': 'test_002',
        'word_count': 6,
        'mode': 'task_prompt',
        'channel_details': {
            'mode': 'task_prompt',
            'triggering_rule': 'primary_red_single_channel_demoted',
            'fusion_counts': {
                'n_primary_red': 1,
                'n_primary_amber': 0,
                'n_primary_yellow_plus': 1,
                'n_yellow_plus': 1,
                'n_red': 1,
                'n_amber_plus': 0,
            },
            'active_channels': 1,
            'short_text_adjustment': False,
            'disabled_channels': [],
            'channels': {
                'prompt_structure': {
                    'score': 0.10,
                    'severity': 'GREEN',
                    'explanation': 'Prompt Structure: no signals',
                    'mode_eligible': True,
                    'data_sufficient': True,
                    'disabled': False,
                    'role': 'primary',
                },
                'continuation': {
                    'score': 0.80,
                    'severity': 'RED',
                    'explanation': 'Continuation: BScore=0.712(API,RED)',
                    'mode_eligible': True,
                    'data_sufficient': True,
                    'disabled': False,
                    'role': 'primary',
                },
                'stylometry': {
                    'score': 0.0,
                    'severity': 'GREEN',
                    'explanation': 'Stylometry: no signals',
                    'mode_eligible': False,
                    'data_sufficient': False,
                    'disabled': False,
                    'role': 'supporting',
                },
                'windowing': {
                    'score': 0.0,
                    'severity': 'GREEN',
                    'explanation': 'Windowing: insufficient text for windows',
                    'mode_eligible': False,
                    'data_sufficient': False,
                    'disabled': False,
                    'role': 'supporting',
                },
            },
        },
    }
    html_out = generate_html_report(text, result)

    check("score bar present", 'score-bar-fill' in html_out, "missing score bar")
    check("numeric score shown", '0.80' in html_out, "missing score value")
    check("Primary role badge shown", 'role-primary' in html_out, "missing primary badge")
    check("Supporting role badge shown", 'role-supporting' in html_out, "missing supporting badge")
    check("No Data badge shown for windowing channel (data_sufficient=False)",
          'role-nodata' in html_out, "missing no-data badge")
    check("Determination Basis section present", 'Determination Basis' in html_out)
    check("Triggering rule displayed",
          'Single primary RED' in html_out or 'primary_red_single_channel_demoted' in html_out,
          "triggering rule not shown")
    check("Fusion count items present", 'Primary RED channels' in html_out)
    check("Continuation (DNA-GPT) label used",
          'Continuation (DNA-GPT)' in html_out, "friendly channel name missing")
    check("Channel table header present", '<th>Score</th>' in html_out)


def test_generate_html_report_disabled_channel():
    print("\n-- GENERATE HTML REPORT: disabled channel badge --")
    text = "Some text."
    result = {
        'detection_spans': [],
        'determination': 'GREEN',
        'reason': 'No significant signals',
        'confidence': 0.0,
        'task_id': 'test_003',
        'word_count': 2,
        'mode': 'task_prompt',
        'channel_details': {
            'mode': 'task_prompt',
            'triggering_rule': 'no_signal',
            'fusion_counts': {
                'n_primary_red': 0, 'n_primary_amber': 0,
                'n_primary_yellow_plus': 0, 'n_yellow_plus': 0,
                'n_red': 0, 'n_amber_plus': 0,
            },
            'active_channels': 0,
            'short_text_adjustment': False,
            'disabled_channels': ['continuation'],
            'channels': {
                'continuation': {
                    'score': 0.0,
                    'severity': 'GREEN',
                    'explanation': 'continuation disabled (ablation)',
                    'mode_eligible': True,
                    'data_sufficient': True,
                    'disabled': True,
                    'role': 'primary',
                },
            },
        },
    }
    html_out = generate_html_report(text, result)
    check("Disabled badge shown", 'role-disabled' in html_out, "missing disabled badge")
    check("No significant signals rule label shown (human-readable)",
          'No significant signals' in html_out,
          "expected human-readable rule label 'No significant signals → GREEN' in output")


if __name__ == '__main__':
    print("=" * 70)
    print("  HTML REPORT GENERATOR TESTS")
    print("=" * 70)

    test_apply_highlights_basic()
    test_apply_highlights_overlapping()
    test_apply_highlights_empty()
    test_generate_html_report_returns_string()
    test_generate_html_report_channel_score_and_role()
    test_generate_html_report_disabled_channel()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
