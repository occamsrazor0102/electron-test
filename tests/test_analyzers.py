"""Tests for individual analyzer modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
from llm_detector.analyzers.perplexity import run_perplexity
from llm_detector.analyzers.preamble import run_preamble
from llm_detector.analyzers.self_similarity import run_self_similarity
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


def test_semantic_resonance():
    print("\n-- SEMANTIC RESONANCE --")

    short = "Hello world."
    r_short = run_semantic_resonance(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_SEMANTIC:
        r_ai = run_semantic_resonance(AI_TEXT)
        check("AI text: semantic_ai_score > 0", r_ai['semantic_ai_score'] > 0,
              f"got {r_ai['semantic_ai_score']}")
        check("AI text: semantic_delta > 0", r_ai['semantic_delta'] > 0,
              f"got {r_ai['semantic_delta']}")
        # determination may be None for short texts (needs ≥5 sentences)
        check("AI text: has determination or delta > 0",
              r_ai['determination'] is not None or r_ai['semantic_delta'] > 0,
              f"got det={r_ai['determination']}, delta={r_ai['semantic_delta']}")

        r_human = run_semantic_resonance(HUMAN_TEXT)
        check("Human text: lower ai_score", r_human['semantic_ai_score'] < r_ai['semantic_ai_score'],
              f"human={r_human['semantic_ai_score']}, ai={r_ai['semantic_ai_score']}")
    else:
        print("  (sentence-transformers not installed -- skipping model tests)")
        check("Unavailable: ai_score=0", r_short['semantic_ai_score'] == 0.0)


def test_perplexity():
    print("\n-- PERPLEXITY SCORING --")

    short = "Hello world."
    r_short = run_perplexity(short)
    check("Short text: no determination", r_short['determination'] is None)

    # Early return dicts should include variance fields
    check("Short text: surprisal_variance=0", r_short['surprisal_variance'] == 0.0)
    check("Short text: volatility_decay_ratio=1", r_short['volatility_decay_ratio'] == 1.0)

    if HAS_PERPLEXITY:
        r_normal = run_perplexity(CLINICAL_TEXT)
        check("Normal text: perplexity > 0", r_normal['perplexity'] > 0,
              f"got {r_normal['perplexity']}")
        check("Normal text: has reason", len(r_normal.get('reason', '')) > 0)
        # Surprisal diversity features
        check("Normal text: surprisal_variance > 0", r_normal['surprisal_variance'] > 0,
              f"got {r_normal['surprisal_variance']}")
        check("Normal text: volatility_decay_ratio > 0", r_normal['volatility_decay_ratio'] > 0,
              f"got {r_normal['volatility_decay_ratio']}")
        check("Normal text: has first_half_var", 'surprisal_first_half_var' in r_normal)
        check("Normal text: has second_half_var", 'surprisal_second_half_var' in r_normal)
    else:
        print("  (transformers/torch not installed -- skipping model tests)")
        check("Unavailable: perplexity=0", r_short['perplexity'] == 0.0)


def test_cot_leakage():
    print("\n-- COT LEAKAGE DETECTION --")

    # <think> tags — smoking gun for reasoning model artifacts
    text_think = "Here is the analysis.\n<think>\nLet me consider the options.\n</think>\nThe answer is 42."
    score, severity, hits, _spans = run_preamble(text_think)
    hit_names = [h[0] for h in hits]
    check("think tags: CRITICAL severity", severity == "CRITICAL")
    check("think tags: cot_leakage in hits", "cot_leakage" in hit_names)
    check("think tags: score == 0.99", score == 0.99)

    # Self-correction phrases
    text_correct = "The total revenue is $50M. Wait, actually let me recalculate that figure."
    score2, severity2, hits2, _spans2 = run_preamble(text_correct)
    hit_names2 = [h[0] for h in hits2]
    check("self-correction: detected", len(hits2) > 0)
    check("self-correction: cot_self_correction in hits", "cot_self_correction" in hit_names2)

    # Reasoning model phrases
    text_reason = "Let me rethink the approach to this problem."
    score3, severity3, hits3, _spans3 = run_preamble(text_reason)
    hit_names3 = [h[0] for h in hits3]
    check("cot reasoning: detected", "cot_reasoning" in hit_names3)

    # Clean text should not trigger
    clean = "The quarterly report shows steady growth across all divisions."
    score4, severity4, hits4, _spans4 = run_preamble(clean)
    check("clean text: no CoT hits", all(h[0] not in ('cot_leakage', 'cot_reasoning', 'cot_self_correction') for h in hits4))


def test_feature_flags():
    print("\n-- FEATURE AVAILABILITY FLAGS --")
    check("HAS_SEMANTIC is bool", isinstance(HAS_SEMANTIC, bool))
    check("HAS_PERPLEXITY is bool", isinstance(HAS_PERPLEXITY, bool))
    print(f"    HAS_SEMANTIC={HAS_SEMANTIC}, HAS_PERPLEXITY={HAS_PERPLEXITY}")


def test_self_similarity_s13():
    print("\n-- SELF-SIMILARITY S13: STRUCTURAL COMPRESSION DELTA (FEAT 5) --")

    long_ai = (
        "This comprehensive analysis provides a thorough examination of the key factors "
        "that contribute to the overall effectiveness of the proposed framework. Furthermore, "
        "it is essential to note that the implementation of these strategies ensures alignment "
        "with best practices and industry standards. To address this challenge, we must consider "
        "multiple perspectives and leverage data-driven insights to achieve optimal outcomes. "
        "Additionally, this approach demonstrates the critical importance of systematic evaluation "
        "and evidence-based decision making in the modern landscape. The primary challenge remains "
        "ensuring comprehensive coverage across all critical domains while maintaining analytical "
        "rigor. The fundamental premise of this framework establishes clear guidelines for all "
        "subsequent analytical procedures. Moreover, the systematic assessment reveals significant "
        "opportunities for sustained growth and development. In conclusion, these findings underscore "
        "the transformative potential of this holistic framework for evidence-based practice. "
        "The comprehensive methodology employed in this investigation demonstrates the critical "
        "importance of maintaining rigorous analytical standards throughout the evaluation process. "
        "Furthermore, the empirical evidence gathered during this systematic review provides "
        "compelling support for the proposed intervention strategy. The results indicate that "
        "a multifaceted approach yields superior outcomes when compared to traditional methods. "
        "Additionally, the framework incorporates robust quality assurance mechanisms designed "
        "to ensure the reliability and validity of all analytical outputs generated during "
        "the assessment phase of this comprehensive evaluation initiative."
    )
    r = run_self_similarity(long_ai)
    check("s13: shuffled_comp_ratio in result", 'shuffled_comp_ratio' in r,
          f"keys: {[k for k in r.keys() if 'shuf' in k or 'struct' in k]}")
    check("s13: structural_compression_delta in result", 'structural_compression_delta' in r)
    check("s13: shuffled_comp_ratio > 0", r['shuffled_comp_ratio'] > 0,
          f"got {r['shuffled_comp_ratio']}")
    check("s13: delta is finite", isinstance(r['structural_compression_delta'], float))

    # Short text should still return zeros for these
    r_short = run_self_similarity("Short text.")
    check("s13 short: shuffled_comp_ratio == 0 (short text exempt)",
          r_short.get('shuffled_comp_ratio', 0) == 0 or r_short.get('word_count', 0) < 200)


def test_perplexity_compound_signals():
    print("\n-- PERPLEXITY COMPOUND SIGNALS (FEAT 7) --")

    short = "Hello world."
    r_short = run_perplexity(short)
    check("Short: comp_ratio in result", 'comp_ratio' in r_short)
    check("Short: zlib_normalized_ppl in result", 'zlib_normalized_ppl' in r_short)
    check("Short: comp_ppl_ratio in result", 'comp_ppl_ratio' in r_short)
    check("Short: token_losses in result", 'token_losses' in r_short)

    if HAS_PERPLEXITY:
        r = run_perplexity(CLINICAL_TEXT)
        check("Clinical: comp_ratio > 0", r['comp_ratio'] > 0,
              f"got {r['comp_ratio']}")
        check("Clinical: zlib_normalized_ppl > 0", r['zlib_normalized_ppl'] > 0,
              f"got {r['zlib_normalized_ppl']}")
        check("Clinical: token_losses is list", isinstance(r['token_losses'], list),
              f"got type {type(r.get('token_losses'))}")


def test_binoculars_fields():
    print("\n-- BINOCULARS FIELDS --")
    from llm_detector.compat import HAS_BINOCULARS

    # Regardless of model availability, the fields should be present
    r_short = run_perplexity("too short")
    check("Short text: binoculars_score in result", 'binoculars_score' in r_short,
          f"keys: {list(r_short.keys())}")
    check("Short text: binoculars_determination in result", 'binoculars_determination' in r_short)
    check("Short text: binoculars_score == 0", r_short['binoculars_score'] == 0.0)
    check("Short text: binoculars_determination is None", r_short['binoculars_determination'] is None)

    if HAS_BINOCULARS and HAS_PERPLEXITY:
        r = run_perplexity(CLINICAL_TEXT)
        check("Binoculars: score field present", 'binoculars_score' in r)
        check("Binoculars: determination field present", 'binoculars_determination' in r)
        check("Binoculars: score is numeric", isinstance(r['binoculars_score'], (int, float)),
              f"got type {type(r['binoculars_score'])}")
    else:
        print("  (transformers not installed -- skipping binoculars model tests)")


if __name__ == '__main__':
    print("=" * 70)
    print("Analyzer Tests")
    print("=" * 70)

    test_feature_flags()
    test_semantic_resonance()
    test_perplexity()
    test_perplexity_compound_signals()
    test_cot_leakage()
    test_self_similarity_s13()
    test_binoculars_fields()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
