"""Tests for Span-Level Explainability (X-Ray View)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.lexicon.packs import score_pack, score_pack_spans, score_all_pack_spans
from llm_detector.analyzers.preamble import run_preamble, run_preamble_spans
from llm_detector.analyzers.fingerprint import run_fingerprint_spans
from llm_detector.analyzers.windowing import get_hot_window_spans
from llm_detector.text_utils import get_sentence_spans
from llm_detector.pipeline import analyze_prompt
from tests.conftest import AI_TEXT, HUMAN_TEXT, CLINICAL_TEXT

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


def test_packscore_spans_field():
    """FEAT 1: PackScore dataclass has spans populated by score_pack()."""
    print("\n-- PACKSCORE.SPANS FIELD --")
    text = "You MUST include all required fields. This is mandatory."
    ps = score_pack(text, 'obligation')
    check("PackScore has spans attr", hasattr(ps, 'spans'))
    check("PackScore.spans is list", isinstance(ps.spans, list))
    check("PackScore.spans non-empty", len(ps.spans) > 0, f"got {len(ps.spans)}")

    for s in ps.spans[:2]:
        check("span is dict", isinstance(s, dict))
        check("span has 'start'", 'start' in s)
        check("span has 'end'", 'end' in s)
        check("span has 'text'", 'text' in s)
        check("span has 'pack'", 'pack' in s)
        check("span has 'weight'", 'weight' in s)
        check(f"span start < end: {s['start']} < {s['end']}",
              s['start'] < s['end'])
        check(f"span text matches source: '{s['text']}'",
              text[s['start']:s['end']] == s['text'] or s['text'] == text[s['start']:s['end']][:80],
              f"text[{s['start']}:{s['end']}]='{text[s['start']:s['end']]}'")
        break


def test_preamble_4tuple():
    """FEAT 1: run_preamble returns 4-tuple with spans."""
    print("\n-- PREAMBLE 4-TUPLE --")
    text = "Sure thing! Here is your updated prompt for the evaluation task."
    result = run_preamble(text)
    check("run_preamble returns 4-tuple", len(result) == 4, f"got {len(result)}")
    score, severity, hits, spans = result
    check("score is float", isinstance(score, (int, float)))
    check("severity is str", isinstance(severity, str))
    check("hits is list", isinstance(hits, list))
    check("spans is list", isinstance(spans, list))
    check("spans non-empty for preamble text", len(spans) > 0, f"got {len(spans)}")

    for s in spans:
        check("preamble span is dict", isinstance(s, dict))
        check("preamble span has start/end", 'start' in s and 'end' in s)
        break

    # Clean text should have empty spans
    clean_result = run_preamble("The quarterly report shows growth.")
    check("clean text: 4-tuple", len(clean_result) == 4)
    check("clean text: empty spans", len(clean_result[3]) == 0)


def test_pipeline_detection_spans():
    """FEAT 2: Pipeline result has 'detection_spans' key."""
    print("\n-- PIPELINE DETECTION_SPANS --")
    text = ("You MUST include all required fields. This comprehensive analysis "
            "provides a thorough examination of the key factors.")
    result = analyze_prompt(text, run_l3=False)
    check("result has 'detection_spans'", 'detection_spans' in result)
    check("detection_spans is list", isinstance(result['detection_spans'], list))
    # Should have some spans for this AI-like text
    n = len(result['detection_spans'])
    check("detection_spans non-empty", n > 0, f"got {n}")
    # Each span should have start key
    for s in result['detection_spans'][:3]:
        check("pipeline span has 'start'", 'start' in s)
        break


def test_pack_spans_basic():
    print("\n-- PACK SPANS: basic structure --")
    text = "You MUST include all required fields. This is mandatory."
    spans = score_pack_spans(text, 'obligation')
    check("obligation spans: non-empty", len(spans) > 0, f"got {len(spans)}")

    for sp in spans:
        check(f"span tuple has 5 elements: {sp[3]}", len(sp) == 5)
        start, end, matched, pack_name, weight = sp
        check(f"start < end: {start} < {end}", start < end, f"{start} >= {end}")
        check(f"matched text matches source: '{matched}'",
              text[start:end] == matched,
              f"text[{start}:{end}]='{text[start:end]}' != '{matched}'")
        check(f"pack_name == 'obligation'", pack_name == 'obligation')
        check(f"weight >= 0", weight >= 0)
        break  # Just check first span in detail


def test_pack_spans_sorted():
    print("\n-- PACK SPANS: sorted by position --")
    text = ("You must include the required fields. Each item must have a valid ID. "
            "The output shall be formatted as JSON.")
    spans = score_pack_spans(text, 'obligation')
    if len(spans) >= 2:
        positions = [s[0] for s in spans]
        check("spans sorted by start_char", positions == sorted(positions),
              f"positions: {positions}")
    else:
        check("enough spans to check order", False, f"only {len(spans)} spans")


def test_all_pack_spans():
    print("\n-- ALL PACK SPANS: multi-pack merge --")
    text = ("You MUST process the CSV file. Each row must have a valid patient_id field. "
            "If the field is null, leave blank. The schema must include patient_id (string, required). "
            "Given the input has a header row, validate all required fields.")
    spans = score_all_pack_spans(text)
    check("all_pack_spans: non-empty", len(spans) > 0, f"got {len(spans)}")

    pack_names = set(s[3] for s in spans)
    check("multiple packs fired", len(pack_names) >= 2,
          f"packs: {pack_names}")

    positions = [s[0] for s in spans]
    check("merged spans sorted", positions == sorted(positions))


def test_preamble_spans():
    print("\n-- PREAMBLE SPANS --")
    text = "Sure thing! Here is your updated prompt for the evaluation task."
    spans = run_preamble_spans(text)
    check("preamble spans: non-empty", len(spans) > 0, f"got {len(spans)}")

    for sp in spans:
        start, end, matched, name, sev = sp
        check(f"preamble span valid: {name}", start < end)
        check(f"matched text correct: '{matched}'",
              text[start:end] == matched,
              f"text[{start}:{end}]='{text[start:end]}' != '{matched}'")
        break

    # CoT leakage
    cot_text = "Let me think.\n<think>\nThis is reasoning.\n</think>\nThe answer."
    cot_spans = run_preamble_spans(cot_text)
    cot_names = [s[3] for s in cot_spans]
    check("CoT leakage spans detected", 'cot_leakage' in cot_names,
          f"names: {cot_names}")

    # Clean text
    clean = "The quarterly report shows growth."
    clean_spans = run_preamble_spans(clean)
    check("clean text: no preamble spans", len(clean_spans) == 0,
          f"got {len(clean_spans)}")


def test_fingerprint_spans():
    print("\n-- FINGERPRINT SPANS --")
    text = ("This comprehensive analysis delves into the nuanced landscape "
            "of leveraging robust frameworks to facilitate holistic outcomes.")
    spans = run_fingerprint_spans(text)
    check("fingerprint spans: non-empty", len(spans) > 0, f"got {len(spans)}")

    words_found = [s[4] for s in spans]
    check("'comprehensive' detected", 'comprehensive' in words_found,
          f"words: {words_found}")
    check("'landscape' detected", 'landscape' in words_found,
          f"words: {words_found}")

    for sp in spans:
        start, end, matched, source, word = sp
        check(f"fingerprint span valid: '{matched}'",
              text[start:end].lower() == matched.lower(),
              f"text[{start}:{end}]='{text[start:end]}' != '{matched}'")
        break

    # Human text should have fewer
    human_spans = run_fingerprint_spans(HUMAN_TEXT)
    check("human text: fewer fingerprint spans",
          len(human_spans) < len(spans),
          f"human={len(human_spans)}, ai={len(spans)}")


def test_sentence_spans():
    print("\n-- SENTENCE SPANS --")
    text = "First sentence. Second sentence. Third sentence here."
    spans = get_sentence_spans(text)
    check("sentence spans: 3 sentences", len(spans) == 3,
          f"got {len(spans)}: {[s[0] for s in spans]}")

    for sent_text, start, end in spans:
        check(f"sentence span covers text: '{sent_text[:30]}'",
              text[start:end] == sent_text,
              f"text[{start}:{end}]='{text[start:end]}' != '{sent_text}'")


def test_hot_window_spans():
    print("\n-- HOT WINDOW SPANS --")
    # AI text should produce windows; need enough sentences
    long_ai = (
        "This comprehensive analysis provides a thorough examination of the key factors. "
        "Furthermore, it is essential to note that the implementation ensures alignment. "
        "To address this challenge, we must consider multiple perspectives. "
        "Additionally, this approach demonstrates the critical importance of evaluation. "
        "The fundamental premise establishes clear guidelines for analytical procedures. "
        "Moreover, the systematic assessment reveals significant opportunities. "
        "In conclusion, these findings underscore the transformative potential. "
        "The methodology employed demonstrates the critical importance of standards. "
        "Furthermore, the empirical evidence provides compelling support. "
        "The results indicate that a multifaceted approach yields superior outcomes."
    )
    spans = get_hot_window_spans(long_ai, threshold=0.10)
    # May or may not have hot windows depending on thresholds
    check("hot_window_spans returns list", isinstance(spans, list))

    for sp in spans:
        check(f"hot window span has 5 elements", len(sp) == 5)
        start, end, score, source, idx = sp
        check(f"window span: start < end ({start} < {end})", start < end)
        check(f"window span: score > 0", score > 0)
        check(f"window span: source == 'hot_window'", source == 'hot_window')
        break


def test_xray_span_positions_accurate():
    print("\n-- SPAN POSITION ACCURACY --")
    text = "The output must include exactly 5 columns and must have valid headers."
    spans = score_pack_spans(text, 'obligation')
    for start, end, matched, pack, weight in spans:
        actual = text[start:end]
        check(f"position accurate: '{matched}' == '{actual}'",
              actual == matched,
              f"at [{start}:{end}] got '{actual}' expected '{matched}'")


def test_empty_text():
    print("\n-- EMPTY/SHORT TEXT --")
    check("empty pack spans", score_pack_spans("", 'obligation') == [])
    check("empty preamble spans", run_preamble_spans("") == [])
    check("empty fingerprint spans", run_fingerprint_spans("") == [])
    check("short hot windows", get_hot_window_spans("Short.") == [])


if __name__ == '__main__':
    print("=" * 70)
    print("  SPAN-LEVEL EXPLAINABILITY (X-RAY) TESTS")
    print("=" * 70)

    test_packscore_spans_field()
    test_preamble_4tuple()
    test_pipeline_detection_spans()
    test_pack_spans_basic()
    test_pack_spans_sorted()
    test_all_pack_spans()
    test_preamble_spans()
    test_fingerprint_spans()
    test_sentence_spans()
    test_hot_window_spans()
    test_xray_span_positions_accurate()
    test_empty_text()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
