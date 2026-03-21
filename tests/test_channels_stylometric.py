"""Comprehensive tests for channels/stylometric.py.

Tests focus on NSSI severity paths, semantic/perplexity boosting, and supporting signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.channels.stylometric import score_stylometric

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


def test_nssi_severity_paths():
    """Test NSSI RED/AMBER/YELLOW severity paths."""
    print("\n-- NSSI SEVERITY PATHS --")

    # NSSI RED
    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.82,
        'nssi_signals': 7,
        'confidence': 0.85
    }
    r = score_stylometric(0, nssi_red)
    check("NSSI RED -> RED severity", r.severity == 'RED')
    check("NSSI RED -> score capped at 0.85", r.score == 0.85)
    check("NSSI score in sub_signals", r.sub_signals['nssi_score'] == 0.82)
    check("NSSI signals count in sub_signals", r.sub_signals['nssi_signals'] == 7)

    # NSSI AMBER
    nssi_amber = {
        'determination': 'AMBER',
        'nssi_score': 0.60,
        'nssi_signals': 4,
        'confidence': 0.65
    }
    r = score_stylometric(0, nssi_amber)
    check("NSSI AMBER -> AMBER severity", r.severity == 'AMBER')
    check("NSSI AMBER -> score capped at 0.65", r.score == 0.65)

    # NSSI YELLOW
    nssi_yellow = {
        'determination': 'YELLOW',
        'nssi_score': 0.35,
        'nssi_signals': 2,
        'confidence': 0.35
    }
    r = score_stylometric(0, nssi_yellow)
    check("NSSI YELLOW -> YELLOW severity", r.severity == 'YELLOW')
    check("NSSI YELLOW -> score capped at 0.40", r.score <= 0.40)


def test_semantic_resonance_standalone():
    """Test semantic resonance as standalone signal."""
    print("\n-- SEMANTIC RESONANCE STANDALONE --")

    # Semantic AMBER standalone
    sem_amber = {
        'determination': 'AMBER',
        'semantic_ai_mean': 0.72,
        'semantic_delta': 0.25,
        'confidence': 0.60
    }
    r = score_stylometric(0, None, semantic=sem_amber)
    check("Semantic AMBER alone -> AMBER severity", r.severity == 'AMBER')
    check("Semantic AMBER -> score from confidence", r.score == 0.60)
    check("Semantic delta in sub_signals", r.sub_signals['semantic_delta'] == 0.25)

    # Semantic YELLOW standalone
    sem_yellow = {
        'determination': 'YELLOW',
        'semantic_ai_mean': 0.58,
        'semantic_delta': 0.15,
        'confidence': 0.35
    }
    r = score_stylometric(0, None, semantic=sem_yellow)
    check("Semantic YELLOW alone -> YELLOW severity", r.severity == 'YELLOW')
    check("Semantic YELLOW -> score from confidence", r.score == 0.35)


def test_semantic_boosting():
    """Test semantic resonance boosting existing signals."""
    print("\n-- SEMANTIC BOOSTING --")

    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.80,
        'nssi_signals': 6,
        'confidence': 0.80
    }
    sem_amber = {
        'determination': 'AMBER',
        'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20,
        'confidence': 0.55
    }

    # NSSI alone
    r_alone = score_stylometric(0, nssi_red)

    # NSSI + Semantic boost
    r_boosted = score_stylometric(0, nssi_red, semantic=sem_amber)
    check("Semantic AMBER boosts NSSI RED", r_boosted.score > r_alone.score)
    check("Semantic boost adds 0.10", abs(r_boosted.score - r_alone.score - 0.10) < 0.01)
    check("Semantic explanation includes 'boost'", 'boost' in r_boosted.explanation.lower())

    # Semantic YELLOW supporting boost
    nssi_amber = {
        'determination': 'AMBER',
        'nssi_score': 0.60,
        'nssi_signals': 4,
        'confidence': 0.60
    }
    sem_yellow = {
        'determination': 'YELLOW',
        'semantic_ai_mean': 0.55,
        'semantic_delta': 0.10,
        'confidence': 0.30
    }
    r_alone = score_stylometric(0, nssi_amber)
    r_boosted = score_stylometric(0, nssi_amber, semantic=sem_yellow)
    check("Semantic YELLOW supports NSSI AMBER", r_boosted.score > r_alone.score)
    check("Semantic YELLOW boost adds 0.05", abs(r_boosted.score - r_alone.score - 0.05) < 0.01)


def test_perplexity_standalone():
    """Test perplexity as standalone signal."""
    print("\n-- PERPLEXITY STANDALONE --")

    # PPL AMBER standalone
    ppl_amber = {
        'determination': 'AMBER',
        'perplexity': 18.5,
        'confidence': 0.58
    }
    r = score_stylometric(0, None, ppl=ppl_amber)
    check("PPL AMBER alone -> AMBER severity", r.severity == 'AMBER')
    check("PPL AMBER -> score from confidence", r.score == 0.58)
    check("Perplexity value in sub_signals", r.sub_signals['perplexity'] == 18.5)

    # PPL YELLOW standalone
    ppl_yellow = {
        'determination': 'YELLOW',
        'perplexity': 25.0,
        'confidence': 0.32
    }
    r = score_stylometric(0, None, ppl=ppl_yellow)
    check("PPL YELLOW alone -> YELLOW severity", r.severity == 'YELLOW')
    check("PPL YELLOW -> score from confidence", r.score == 0.32)


def test_perplexity_boosting():
    """Test perplexity boosting existing signals."""
    print("\n-- PERPLEXITY BOOSTING --")

    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.78,
        'nssi_signals': 6,
        'confidence': 0.78
    }
    ppl_amber = {
        'determination': 'AMBER',
        'perplexity': 16.2,
        'confidence': 0.55
    }

    # NSSI alone vs NSSI + PPL
    r_alone = score_stylometric(0, nssi_red)
    r_boosted = score_stylometric(0, nssi_red, ppl=ppl_amber)
    check("PPL AMBER boosts NSSI RED", r_boosted.score > r_alone.score)
    check("PPL boost adds 0.10", abs(r_boosted.score - r_alone.score - 0.10) < 0.01)


def test_surprisal_variance_boosting():
    """Test surprisal variance and volatility decay boosting."""
    print("\n-- SURPRISAL VARIANCE BOOSTING --")

    nssi_amber = {
        'determination': 'AMBER',
        'nssi_score': 0.60,
        'nssi_signals': 4,
        'confidence': 0.60
    }

    # High boost: variance < 2.0 and decay > 1.5
    ppl_high = {
        'surprisal_variance': 1.5,
        'volatility_decay_ratio': 1.8
    }
    r = score_stylometric(0, nssi_amber, ppl=ppl_high)
    check("Low variance + high decay -> boosts score", r.score > 0.60)
    check("High variance boost adds ~0.08", abs(r.score - 0.60 - 0.08) < 0.02)
    check("Variance in sub_signals", 'surprisal_variance' in r.sub_signals)
    check("Decay ratio in sub_signals", 'volatility_decay_ratio' in r.sub_signals)

    # Mild boost: variance < 3.0 and decay > 1.2
    ppl_mild = {
        'surprisal_variance': 2.5,
        'volatility_decay_ratio': 1.3
    }
    r = score_stylometric(0, nssi_amber, ppl=ppl_mild)
    check("Mild variance/decay -> mild boost", r.score > 0.60)
    check("Mild boost adds ~0.04", abs(r.score - 0.60 - 0.04) < 0.02)

    # No boost: variance too high or decay too low
    ppl_none = {
        'surprisal_variance': 4.0,
        'volatility_decay_ratio': 1.0
    }
    r = score_stylometric(0, nssi_amber, ppl=ppl_none)
    check("High variance/low decay -> no boost", r.score == 0.60)


def test_tocsin_corroboration():
    """Test TOCSIN (token cohesiveness) corroboration."""
    print("\n-- TOCSIN CORROBORATION --")

    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.80,
        'nssi_signals': 6,
        'confidence': 0.80
    }

    # TOCSIN AMBER boost
    tocsin_amber = {
        'determination': 'AMBER',
        'cohesiveness': 0.8234,
        'confidence': 0.52
    }
    r_alone = score_stylometric(0, nssi_red)
    r_boosted = score_stylometric(0, nssi_red, tocsin=tocsin_amber)
    check("TOCSIN AMBER boosts NSSI RED", r_boosted.score > r_alone.score)
    check("TOCSIN boost adds 0.10", abs(r_boosted.score - r_alone.score - 0.10) < 0.01)
    check("Cohesiveness in sub_signals", 'cohesiveness' in r_boosted.sub_signals)

    # TOCSIN AMBER standalone
    r = score_stylometric(0, None, tocsin=tocsin_amber)
    check("TOCSIN AMBER alone -> AMBER severity", r.severity == 'AMBER')
    check("TOCSIN AMBER -> score from confidence", r.score == 0.52)

    # TOCSIN YELLOW supporting
    tocsin_yellow = {
        'determination': 'YELLOW',
        'cohesiveness': 0.7651,
        'confidence': 0.30
    }
    nssi_amber = {
        'determination': 'AMBER',
        'nssi_score': 0.60,
        'nssi_signals': 4,
        'confidence': 0.60
    }
    r = score_stylometric(0, nssi_amber, tocsin=tocsin_yellow)
    check("TOCSIN YELLOW supports", r.score > 0.60)
    check("TOCSIN YELLOW adds 0.05", abs(r.score - 0.60 - 0.05) < 0.01)


def test_binoculars_signal():
    """Test binoculars contrastive LM ratio signal."""
    print("\n-- BINOCULARS SIGNAL --")

    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.80,
        'nssi_signals': 6,
        'confidence': 0.80
    }

    # Binoculars AMBER boost
    ppl_bino = {
        'binoculars_determination': 'AMBER',
        'binoculars_score': 0.45,
        'confidence': 0.55
    }
    r_alone = score_stylometric(0, nssi_red)
    r_boosted = score_stylometric(0, nssi_red, ppl=ppl_bino)
    check("Binoculars AMBER boosts NSSI RED", r_boosted.score > r_alone.score)
    check("Binoculars in sub_signals", 'binoculars_score' in r_boosted.sub_signals)

    # Binoculars AMBER standalone
    r = score_stylometric(0, None, ppl=ppl_bino)
    check("Binoculars AMBER alone -> AMBER severity", r.severity == 'AMBER')

    # Binoculars YELLOW supporting
    ppl_bino_yellow = {
        'binoculars_determination': 'YELLOW',
        'binoculars_score': 0.30,
        'confidence': 0.30
    }
    nssi_amber = {
        'determination': 'AMBER',
        'nssi_score': 0.60,
        'nssi_signals': 4,
        'confidence': 0.60
    }
    r = score_stylometric(0, nssi_amber, ppl=ppl_bino_yellow)
    check("Binoculars YELLOW supports", r.score > 0.60)


def test_fingerprint_supporting_weight():
    """Test that fingerprints add supporting weight when other signals active."""
    print("\n-- FINGERPRINT SUPPORTING WEIGHT --")

    nssi_amber = {
        'determination': 'AMBER',
        'nssi_score': 0.60,
        'nssi_signals': 4,
        'confidence': 0.60
    }

    # Fingerprint >= 0.30 adds supporting weight
    r_alone = score_stylometric(0, nssi_amber)
    r_with_fp = score_stylometric(0.35, nssi_amber)
    check("Fingerprint >= 0.30 adds weight", r_with_fp.score > r_alone.score)
    check("Fingerprint adds 0.10", abs(r_with_fp.score - r_alone.score - 0.10) < 0.01)
    check("Fingerprint in sub_signals", 'fingerprints' in r_with_fp.sub_signals)
    check("Fingerprint explanation", 'supporting' in r_with_fp.explanation.lower())

    # Fingerprint < 0.30 does not add weight
    r_low_fp = score_stylometric(0.25, nssi_amber)
    check("Fingerprint < 0.30 does not add weight", r_low_fp.score == r_alone.score)

    # Fingerprint does not boost if severity is GREEN
    r_green = score_stylometric(0.50, None)
    check("Fingerprint does not boost GREEN", r_green.severity == 'GREEN')
    check("Fingerprint does not boost GREEN score", r_green.score == 0)


def test_combined_boosting():
    """Test multiple boosting signals together."""
    print("\n-- COMBINED BOOSTING --")

    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.75,
        'nssi_signals': 6,
        'confidence': 0.75
    }
    sem_amber = {
        'determination': 'AMBER',
        'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20,
        'confidence': 0.55
    }
    ppl_amber = {
        'determination': 'AMBER',
        'perplexity': 18.0,
        'confidence': 0.55,
        'surprisal_variance': 1.8,
        'volatility_decay_ratio': 1.6
    }
    tocsin_amber = {
        'determination': 'AMBER',
        'cohesiveness': 0.8100,
        'confidence': 0.50
    }

    r_alone = score_stylometric(0, nssi_red)
    r_all = score_stylometric(0.40, nssi_red, semantic=sem_amber, ppl=ppl_amber, tocsin=tocsin_amber)

    # Combined signals should boost the score significantly or cap at 1.0
    check("Combined boosting increases score or caps at 1.0",
          r_all.score > r_alone.score or r_all.score == 1.0,
          f"alone={r_alone.score}, combined={r_all.score}")
    check("Combined boosting stays capped at 1.0", r_all.score <= 1.0)
    check("All signals in sub_signals", len(r_all.sub_signals) >= 6)


def test_no_signals_baseline():
    """Test that no signals produce GREEN with zero score."""
    print("\n-- NO SIGNALS BASELINE --")

    r = score_stylometric(0, None)
    check("No signals -> GREEN", r.severity == 'GREEN')
    check("No signals -> score 0", r.score == 0)
    check("No signals -> explanation indicates none", 'no signals' in r.explanation.lower())
    check("No signals -> data_sufficient=False", r.data_sufficient == False)


def test_data_sufficiency():
    """Test data_sufficient flag when analyzers produce results."""
    print("\n-- DATA SUFFICIENCY --")

    # With data
    nssi_red = {
        'determination': 'RED',
        'nssi_score': 0.80,
        'nssi_signals': 6,
        'confidence': 0.80
    }
    r = score_stylometric(0, nssi_red)
    check("With NSSI data -> data_sufficient=True", r.data_sufficient == True)

    # With semantic data only
    sem_amber = {
        'determination': 'AMBER',
        'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20,
        'confidence': 0.55
    }
    r = score_stylometric(0, None, semantic=sem_amber)
    check("With semantic data -> data_sufficient=True", r.data_sufficient == True)

    # No data at all
    r = score_stylometric(0, None)
    check("No data -> data_sufficient=False", r.data_sufficient == False)


if __name__ == '__main__':
    test_nssi_severity_paths()
    test_semantic_resonance_standalone()
    test_semantic_boosting()
    test_perplexity_standalone()
    test_perplexity_boosting()
    test_surprisal_variance_boosting()
    test_tocsin_corroboration()
    test_binoculars_signal()
    test_fingerprint_supporting_weight()
    test_combined_boosting()
    test_no_signals_baseline()
    test_data_sufficiency()

    print(f"\n{'='*60}")
    print(f"TOTAL: {PASSED} passed, {FAILED} failed")
    print(f"{'='*60}")

    if FAILED > 0:
        sys.exit(1)
