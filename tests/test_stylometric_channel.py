"""Tests for channels/stylometric.py — target 85% coverage."""

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


def test_no_signals_green():
    """No inputs → GREEN, data_sufficient=False."""
    print("\n-- NO SIGNALS --")
    r = score_stylometric(fingerprint_score=0, self_sim=None)
    check("No signals -> GREEN", r.severity == 'GREEN', f"got {r.severity}")
    check("No signals -> score 0.0", r.score == 0.0, f"got {r.score}")
    check("data_sufficient=False when empty", r.data_sufficient is False)
    check("Explanation 'no signals'", 'no signals' in r.explanation)
    check("channel name is stylometry", r.channel == 'stylometry')
    check("generic_aigt in mode_eligibility", 'generic_aigt' in r.mode_eligibility)


def test_nssi_red():
    """NSSI determination=RED → RED severity."""
    print("\n-- NSSI RED --")
    self_sim = {'determination': 'RED', 'nssi_score': 0.80, 'nssi_signals': 5,
                'confidence': 0.85}
    r = score_stylometric(0, self_sim)
    check("NSSI RED -> RED", r.severity == 'RED', f"got {r.severity}")
    check("NSSI RED score capped at 0.85", r.score == 0.85, f"got {r.score}")
    check("nssi_score in sub_signals", 'nssi_score' in r.sub_signals)
    check("nssi_signals in sub_signals", 'nssi_signals' in r.sub_signals)
    check("data_sufficient=True", r.data_sufficient is True)

    # Without confidence, score defaults to 0.80
    self_sim_no_conf = {'determination': 'RED', 'nssi_score': 0.80, 'nssi_signals': 5}
    r2 = score_stylometric(0, self_sim_no_conf)
    check("NSSI RED no conf -> score 0.80", r2.score == 0.80, f"got {r2.score}")


def test_nssi_amber():
    """NSSI determination=AMBER → AMBER severity."""
    print("\n-- NSSI AMBER --")
    self_sim = {'determination': 'AMBER', 'nssi_score': 0.55, 'nssi_signals': 3,
                'confidence': 0.65}
    r = score_stylometric(0, self_sim)
    check("NSSI AMBER -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("NSSI AMBER score capped at 0.65", r.score == 0.65, f"got {r.score}")

    # Score capped at 0.65 max
    self_sim_high_conf = {'determination': 'AMBER', 'nssi_score': 0.55, 'nssi_signals': 3,
                          'confidence': 0.90}
    r2 = score_stylometric(0, self_sim_high_conf)
    check("NSSI AMBER score capped at max 0.65", r2.score == 0.65, f"got {r2.score}")


def test_nssi_yellow():
    """NSSI determination=YELLOW → YELLOW severity."""
    print("\n-- NSSI YELLOW --")
    self_sim = {'determination': 'YELLOW', 'nssi_score': 0.30, 'nssi_signals': 2,
                'confidence': 0.30}
    r = score_stylometric(0, self_sim)
    check("NSSI YELLOW -> YELLOW", r.severity == 'YELLOW', f"got {r.severity}")
    check("NSSI YELLOW score <= 0.40", r.score <= 0.40, f"got {r.score}")


def test_semantic_amber_standalone():
    """Semantic determination=AMBER alone → AMBER severity."""
    print("\n-- SEMANTIC AMBER STANDALONE --")
    semantic = {'determination': 'AMBER', 'semantic_ai_mean': 0.70,
                'semantic_delta': 0.20, 'confidence': 0.55}
    r = score_stylometric(0, None, semantic=semantic)
    check("Semantic AMBER -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("semantic_delta in sub_signals", 'semantic_delta' in r.sub_signals)
    check("semantic_ai_score in sub_signals", 'semantic_ai_score' in r.sub_signals)


def test_semantic_yellow_standalone():
    """Semantic determination=YELLOW alone → YELLOW severity."""
    print("\n-- SEMANTIC YELLOW STANDALONE --")
    semantic = {'determination': 'YELLOW', 'semantic_ai_mean': 0.55,
                'semantic_delta': 0.10, 'confidence': 0.30}
    r = score_stylometric(0, None, semantic=semantic)
    check("Semantic YELLOW -> YELLOW", r.severity == 'YELLOW', f"got {r.severity}")
    check("score > 0", r.score > 0, f"got {r.score}")


def test_semantic_amber_boosts_red_nssi():
    """Semantic AMBER boosts score when NSSI already at RED/AMBER."""
    print("\n-- SEMANTIC AMBER BOOSTS NSSI --")
    self_sim_red = {'determination': 'RED', 'nssi_score': 0.80, 'nssi_signals': 5,
                    'confidence': 0.80}
    r_no_sem = score_stylometric(0, self_sim_red)
    base_score = r_no_sem.score

    semantic = {'determination': 'AMBER', 'semantic_ai_mean': 0.70,
                'semantic_delta': 0.20, 'confidence': 0.55}
    r_with_sem = score_stylometric(0, self_sim_red, semantic=semantic)
    check("Semantic AMBER boosts RED NSSI score",
          r_with_sem.score > base_score,
          f"base={base_score}, boosted={r_with_sem.score}")
    check("Severity stays RED", r_with_sem.severity == 'RED')

    # Boost adds 0.10
    check("Score boost == +0.10", abs(r_with_sem.score - (base_score + 0.10)) < 0.001,
          f"expected {base_score + 0.10}, got {r_with_sem.score}")


def test_semantic_yellow_boosts_existing_signal():
    """Semantic YELLOW provides small boost when severity is already non-GREEN."""
    print("\n-- SEMANTIC YELLOW BOOSTS EXISTING --")
    self_sim_amber = {'determination': 'AMBER', 'nssi_score': 0.55, 'nssi_signals': 3,
                      'confidence': 0.60}
    r_no_sem = score_stylometric(0, self_sim_amber)
    base_score = r_no_sem.score

    semantic = {'determination': 'YELLOW', 'semantic_ai_mean': 0.55,
                'semantic_delta': 0.10, 'confidence': 0.30}
    r_with_sem = score_stylometric(0, self_sim_amber, semantic=semantic)
    check("Semantic YELLOW boosts AMBER NSSI",
          r_with_sem.score > base_score,
          f"base={base_score}, boosted={r_with_sem.score}")
    # Score boost is +0.05
    check("Score boost == +0.05",
          abs(r_with_sem.score - (base_score + 0.05)) < 0.001,
          f"expected {base_score + 0.05}, got {r_with_sem.score}")


def test_ppl_amber_standalone():
    """Perplexity determination=AMBER alone → AMBER severity."""
    print("\n-- PPL AMBER STANDALONE --")
    ppl = {'determination': 'AMBER', 'perplexity': 12.0, 'confidence': 0.55}
    r = score_stylometric(0, None, ppl=ppl)
    check("PPL AMBER -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("perplexity in sub_signals", 'perplexity' in r.sub_signals)


def test_ppl_yellow_standalone():
    """Perplexity determination=YELLOW alone → YELLOW severity."""
    print("\n-- PPL YELLOW STANDALONE --")
    ppl = {'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30}
    r = score_stylometric(0, None, ppl=ppl)
    check("PPL YELLOW -> YELLOW", r.severity == 'YELLOW', f"got {r.severity}")


def test_ppl_amber_boosts_existing():
    """PPL AMBER adds +0.10 when NSSI already at RED or AMBER."""
    print("\n-- PPL AMBER BOOSTS NSSI --")
    self_sim = {'determination': 'RED', 'nssi_score': 0.80, 'nssi_signals': 5,
                'confidence': 0.80}
    r_no_ppl = score_stylometric(0, self_sim)
    base = r_no_ppl.score

    ppl = {'determination': 'AMBER', 'perplexity': 10.0, 'confidence': 0.60}
    r = score_stylometric(0, self_sim, ppl=ppl)
    check("PPL AMBER boosts RED NSSI",
          r.score > base, f"base={base}, boosted={r.score}")
    check("PPL boost == +0.10",
          abs(r.score - min(base + 0.10, 1.0)) < 0.001,
          f"expected {min(base + 0.10, 1.0)}, got {r.score}")


def test_ppl_yellow_boosts_existing():
    """PPL YELLOW adds +0.05 when severity is already non-GREEN."""
    print("\n-- PPL YELLOW BOOSTS EXISTING --")
    self_sim_yellow = {'determination': 'YELLOW', 'nssi_score': 0.30, 'nssi_signals': 2,
                       'confidence': 0.30}
    r_no_ppl = score_stylometric(0, self_sim_yellow)
    base = r_no_ppl.score

    ppl = {'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30}
    r = score_stylometric(0, self_sim_yellow, ppl=ppl)
    check("PPL YELLOW boosts YELLOW NSSI",
          r.score > base, f"base={base}, boosted={r.score}")


def test_surprisal_variance_in_sub_signals():
    """Surprisal variance > 0 is stored in sub_signals."""
    print("\n-- SURPRISAL VARIANCE IN SUB_SIGNALS --")
    ppl = {'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
           'surprisal_variance': 3.0, 'volatility_decay_ratio': 1.5}
    r = score_stylometric(0, None, ppl=ppl)
    check("surprisal_variance in sub_signals",
          'surprisal_variance' in r.sub_signals,
          f"got keys: {list(r.sub_signals.keys())}")
    check("volatility_decay_ratio in sub_signals",
          'volatility_decay_ratio' in r.sub_signals)


def test_surprisal_variance_boost():
    """Low surprisal variance + high decay ratio boosts score."""
    print("\n-- SURPRISAL VARIANCE BOOST --")
    # Need non-GREEN severity for boost to apply; use PPL YELLOW to set severity
    # Then check boost from low variance + high decay
    ppl_base = {'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
                'surprisal_variance': 0, 'volatility_decay_ratio': 1.0}
    r_no_boost = score_stylometric(0, None, ppl=ppl_base)

    ppl_boost = {'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
                 'surprisal_variance': 1.5, 'volatility_decay_ratio': 2.0}
    r_boost = score_stylometric(0, None, ppl=ppl_boost)
    check("Low var + high decay boosts score",
          r_boost.score > r_no_boost.score,
          f"no_boost={r_no_boost.score}, boosted={r_boost.score}")

    # Mild boost: s_var < 3.0, v_decay > 1.2
    ppl_mild = {'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
                'surprisal_variance': 2.5, 'volatility_decay_ratio': 1.3}
    r_mild = score_stylometric(0, None, ppl=ppl_mild)
    check("Mild variance boost applied",
          r_mild.score > r_no_boost.score,
          f"no_boost={r_no_boost.score}, mild={r_mild.score}")


def test_tocsin_amber_standalone():
    """TOCSIN determination=AMBER alone → AMBER severity."""
    print("\n-- TOCSIN AMBER STANDALONE --")
    tocsin = {'determination': 'AMBER', 'cohesiveness': 0.025, 'confidence': 0.55}
    r = score_stylometric(0, None, tocsin=tocsin)
    check("TOCSIN AMBER standalone -> AMBER",
          r.severity == 'AMBER', f"got {r.severity}")
    check("cohesiveness in sub_signals", 'cohesiveness' in r.sub_signals)


def test_tocsin_amber_boosts_existing():
    """TOCSIN AMBER adds +0.10 when NSSI is RED/AMBER."""
    print("\n-- TOCSIN AMBER BOOSTS NSSI --")
    self_sim = {'determination': 'AMBER', 'nssi_score': 0.55, 'nssi_signals': 3,
                'confidence': 0.60}
    r_no_tocsin = score_stylometric(0, self_sim)
    base = r_no_tocsin.score

    tocsin = {'determination': 'AMBER', 'cohesiveness': 0.030, 'confidence': 0.55}
    r = score_stylometric(0, self_sim, tocsin=tocsin)
    check("TOCSIN boosts AMBER NSSI",
          r.score > base, f"base={base}, boosted={r.score}")


def test_tocsin_yellow_supporting():
    """TOCSIN YELLOW provides small boost when severity is non-GREEN."""
    print("\n-- TOCSIN YELLOW SUPPORTING --")
    self_sim = {'determination': 'YELLOW', 'nssi_score': 0.30, 'nssi_signals': 2,
                'confidence': 0.30}
    r_no_tocsin = score_stylometric(0, self_sim)
    base = r_no_tocsin.score

    tocsin = {'determination': 'YELLOW', 'cohesiveness': 0.015, 'confidence': 0.25}
    r = score_stylometric(0, self_sim, tocsin=tocsin)
    check("TOCSIN YELLOW boosts YELLOW NSSI",
          r.score > base, f"base={base}, boosted={r.score}")


def test_fingerprint_in_sub_signals():
    """fingerprint_score > 0 is stored in sub_signals."""
    print("\n-- FINGERPRINT IN SUB_SIGNALS --")
    r = score_stylometric(fingerprint_score=0.25, self_sim=None)
    check("fingerprints in sub_signals", 'fingerprints' in r.sub_signals,
          f"got keys: {list(r.sub_signals.keys())}")
    check("fingerprints value correct", r.sub_signals['fingerprints'] == 0.25)


def test_fingerprint_supporting_weight():
    """fingerprint_score >= 0.30 adds +0.10 when severity is non-GREEN."""
    print("\n-- FINGERPRINT SUPPORTING WEIGHT --")
    self_sim = {'determination': 'YELLOW', 'nssi_score': 0.30, 'nssi_signals': 2,
                'confidence': 0.30}
    r_no_fp = score_stylometric(0.0, self_sim)
    base = r_no_fp.score

    r_fp = score_stylometric(0.40, self_sim)
    check("Fingerprint 0.40 adds weight when non-GREEN",
          r_fp.score > base, f"base={base}, with_fp={r_fp.score}")

    # fingerprint < 0.30 should not add supporting weight
    r_fp_low = score_stylometric(0.20, self_sim)
    check("Fingerprint 0.20 does not add extra weight",
          r_fp_low.score == base, f"base={base}, with_low_fp={r_fp_low.score}")


def test_binoculars_amber_standalone():
    """Binoculars determination=AMBER alone → AMBER severity."""
    print("\n-- BINOCULARS AMBER STANDALONE --")
    ppl = {'binoculars_determination': 'AMBER', 'binoculars_score': 0.60,
           'confidence': 0.55, 'surprisal_variance': 0, 'volatility_decay_ratio': 1.0}
    r = score_stylometric(0, None, ppl=ppl)
    check("Binoculars AMBER -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("binoculars_score in sub_signals", 'binoculars_score' in r.sub_signals)


def test_binoculars_yellow_boosts_existing():
    """Binoculars YELLOW adds +0.05 when severity is non-GREEN."""
    print("\n-- BINOCULARS YELLOW BOOSTS EXISTING --")
    self_sim = {'determination': 'YELLOW', 'nssi_score': 0.30, 'nssi_signals': 2,
                'confidence': 0.30}
    r_no_bino = score_stylometric(0, self_sim)
    base = r_no_bino.score

    ppl = {'binoculars_determination': 'YELLOW', 'binoculars_score': 0.80,
           'surprisal_variance': 0, 'volatility_decay_ratio': 1.0}
    r = score_stylometric(0, self_sim, ppl=ppl)
    check("Binoculars YELLOW boosts YELLOW NSSI",
          r.score > base, f"base={base}, boosted={r.score}")


if __name__ == '__main__':
    print("=" * 70)
    print("Stylometric Channel Tests")
    print("=" * 70)

    test_no_signals_green()
    test_nssi_red()
    test_nssi_amber()
    test_nssi_yellow()
    test_semantic_amber_standalone()
    test_semantic_yellow_standalone()
    test_semantic_amber_boosts_red_nssi()
    test_semantic_yellow_boosts_existing_signal()
    test_ppl_amber_standalone()
    test_ppl_yellow_standalone()
    test_ppl_amber_boosts_existing()
    test_ppl_yellow_boosts_existing()
    test_surprisal_variance_in_sub_signals()
    test_surprisal_variance_boost()
    test_tocsin_amber_standalone()
    test_tocsin_amber_boosts_existing()
    test_tocsin_yellow_supporting()
    test_fingerprint_in_sub_signals()
    test_fingerprint_supporting_weight()
    test_binoculars_amber_standalone()
    test_binoculars_yellow_boosts_existing()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:

        sys.exit(1)
