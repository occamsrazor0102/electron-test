"""Tests for channels/prompt_structure.py — target 85% coverage."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.channels.prompt_structure import score_prompt_structure

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


# Minimal voice_dis dict that satisfies all required keys
_VD_NONE = {'voice_gated': False, 'vsd': 0, 'contractions': 0,
            'spec_score': 0, 'voice_score': 0.0, 'hedges': 0}


def test_preamble_critical_instant_red():
    """CRITICAL preamble bypasses all other signals and returns RED immediately."""
    print("\n-- PREAMBLE CRITICAL --")
    r = score_prompt_structure(
        preamble_score=0.99, preamble_severity='CRITICAL',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=50,
    )
    check("CRITICAL preamble -> RED", r.severity == 'RED', f"got {r.severity}")
    check("CRITICAL preamble score 0.99", r.score == 0.99, f"got {r.score}")
    check("CRITICAL preamble explanation contains 'critical hit'",
          'critical hit' in r.explanation.lower())
    check("sub_signals has preamble=0.99", r.sub_signals.get('preamble') == 0.99)


def test_preamble_score_above_threshold():
    """preamble_score >= 0.50 contributes to score even without CRITICAL severity."""
    print("\n-- PREAMBLE SCORE >= 0.50 --")
    r = score_prompt_structure(
        preamble_score=0.60, preamble_severity='AMBER',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("preamble 0.60 in sub_signals", 'preamble' in r.sub_signals)
    check("preamble score >= 0.60", r.score >= 0.60, f"got {r.score}")

    # Below threshold: preamble_score < 0.50 should not contribute
    r2 = score_prompt_structure(
        preamble_score=0.30, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("preamble 0.30 not in sub_signals", 'preamble' not in r2.sub_signals)


def test_prompt_signature_red_threshold():
    """prompt_sig composite >= 0.60 triggers RED."""
    print("\n-- PROMPT SIGNATURE RED --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.65},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("prompt_sig 0.65 -> RED", r.severity == 'RED', f"got {r.severity}")
    check("prompt_sig 0.65 score >= 0.65", r.score >= 0.65, f"got {r.score}")
    check("prompt_sig in sub_signals", 'prompt_signature' in r.sub_signals)
    check("explanation mentions RED", 'RED' in r.explanation)

    # Boundary: exactly 0.60
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.60},
        voice_dis=_VD_NONE, instr_density=None, word_count=100,
    )
    check("prompt_sig 0.60 -> RED", r2.severity == 'RED', f"got {r2.severity}")


def test_prompt_signature_amber_threshold():
    """prompt_sig composite in [0.40, 0.60) triggers AMBER."""
    print("\n-- PROMPT SIGNATURE AMBER --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.45},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("prompt_sig 0.45 -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("prompt_sig 0.45 score >= 0.45", r.score >= 0.45, f"got {r.score}")
    check("explanation mentions AMBER", 'AMBER' in r.explanation)


def test_prompt_signature_yellow_threshold():
    """prompt_sig composite in [0.20, 0.40) triggers YELLOW."""
    print("\n-- PROMPT SIGNATURE YELLOW --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.25},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("prompt_sig 0.25 -> YELLOW", r.severity == 'YELLOW', f"got {r.severity}")
    check("prompt_sig 0.25 score > 0", r.score > 0, f"got {r.score}")
    check("explanation mentions YELLOW", 'YELLOW' in r.explanation)

    # Below threshold: 0.10 should produce GREEN
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.10},
        voice_dis=_VD_NONE, instr_density=None, word_count=100,
    )
    check("prompt_sig 0.10 -> GREEN (no signal)", r2.severity == 'GREEN',
          f"got {r2.severity}")


def test_vsd_gated_red():
    """Voice-gated VSD >= 50 triggers RED."""
    print("\n-- VSD GATED RED --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': True, 'vsd': 55, 'contractions': 0,
                   'spec_score': 0, 'voice_score': 0.0, 'hedges': 0},
        instr_density=None,
        word_count=100,
    )
    check("VSD gated 55 -> RED", r.severity == 'RED', f"got {r.severity}")
    check("VSD gated 55 score == 0.90", r.score == 0.90, f"got {r.score}")
    check("vsd_gated in sub_signals", 'vsd_gated' in r.sub_signals)

    # Boundary: exactly 50
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': True, 'vsd': 50, 'contractions': 0,
                   'spec_score': 0, 'voice_score': 0.0, 'hedges': 0},
        instr_density=None, word_count=100,
    )
    check("VSD gated exactly 50 -> RED", r2.severity == 'RED', f"got {r2.severity}")


def test_vsd_gated_amber():
    """Voice-gated VSD in [21, 50) triggers AMBER."""
    print("\n-- VSD GATED AMBER --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': True, 'vsd': 25, 'contractions': 0,
                   'spec_score': 0, 'voice_score': 0.0, 'hedges': 0},
        instr_density=None,
        word_count=100,
    )
    check("VSD gated 25 -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("VSD gated 25 score == 0.70", r.score == 0.70, f"got {r.score}")


def test_idi_red():
    """Instruction density IDI >= 12 triggers RED."""
    print("\n-- IDI RED --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE,
        instr_density={'idi': 13},
        word_count=100,
    )
    check("IDI 13 -> RED", r.severity == 'RED', f"got {r.severity}")
    check("IDI 13 score == 0.85", r.score == 0.85, f"got {r.score}")
    check("idi in sub_signals", 'idi' in r.sub_signals)

    # Boundary: exactly 12
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE, instr_density={'idi': 12}, word_count=100,
    )
    check("IDI exactly 12 -> RED", r2.severity == 'RED', f"got {r2.severity}")


def test_idi_amber():
    """Instruction density IDI in [8, 12) triggers AMBER."""
    print("\n-- IDI AMBER --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE,
        instr_density={'idi': 9},
        word_count=100,
    )
    check("IDI 9 -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("IDI 9 score == 0.65", r.score == 0.65, f"got {r.score}")


def test_ssi_triggered_yellow():
    """SSI triggered at spec_score in [5, 8) with correct conditions → YELLOW."""
    print("\n-- SSI YELLOW --")
    # contractions=0 → threshold 5.0; spec_score=6.0, voice_score=0.3 < 0.5, hedges=0, wc=200 >= 150
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0,
                   'spec_score': 6.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=200,
    )
    check("SSI triggered contractions=0 spec=6.0 -> YELLOW",
          r.severity == 'YELLOW', f"got {r.severity}")
    check("SSI in explanation", 'SSI' in r.explanation)
    check("ssi in sub_signals", 'ssi' in r.sub_signals)


def test_ssi_triggered_amber():
    """SSI triggered at spec_score >= 8.0 → AMBER."""
    print("\n-- SSI AMBER --")
    # contractions=0 → threshold 5.0; spec_score=9.0 >= 8.0 → AMBER
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0,
                   'spec_score': 9.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=200,
    )
    check("SSI triggered spec=9.0 -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("SSI score >= 0.70", r.score >= 0.70, f"got {r.score}")

    # contractions>0 → threshold 7.0; spec_score=8.0 >= 8.0 → AMBER
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 2,
                   'spec_score': 8.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=200,
    )
    check("SSI triggered contractions=2 spec=8.0 -> AMBER",
          r2.severity == 'AMBER', f"got {r2.severity}")


def test_ssi_not_triggered_word_count():
    """SSI not triggered when word_count < 150."""
    print("\n-- SSI NOT TRIGGERED (word_count < 150) --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0,
                   'spec_score': 6.0, 'voice_score': 0.3, 'hedges': 0},
        instr_density=None,
        word_count=100,  # < 150
    )
    check("SSI not triggered with wc=100", 'SSI' not in r.explanation)
    check("No ssi sub_signal", 'ssi' not in r.sub_signals)


def test_ssi_not_triggered_high_voice_score():
    """SSI not triggered when voice_score >= 0.5."""
    print("\n-- SSI NOT TRIGGERED (high voice_score) --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0,
                   'spec_score': 6.0, 'voice_score': 0.6, 'hedges': 0},
        instr_density=None,
        word_count=200,
    )
    check("SSI not triggered with voice_score=0.6", 'SSI' not in r.explanation)

    # hedges > 0 also prevents SSI
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 0, 'contractions': 0,
                   'spec_score': 6.0, 'voice_score': 0.3, 'hedges': 1},
        instr_density=None,
        word_count=200,
    )
    check("SSI not triggered with hedges=1", 'SSI' not in r2.explanation)


def test_vsd_ungated_amber():
    """Ungated VSD >= 100 triggers AMBER."""
    print("\n-- VSD UNGATED AMBER --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 105, 'contractions': 0,
                   'spec_score': 0, 'voice_score': 0.0, 'hedges': 0},
        instr_density=None,
        word_count=100,
    )
    check("VSD ungated 105 -> AMBER", r.severity == 'AMBER', f"got {r.severity}")
    check("VSD ungated 105 score == 0.60", r.score == 0.60, f"got {r.score}")
    check("vsd_ungated in sub_signals", 'vsd_ungated' in r.sub_signals)

    # Boundary: exactly 100
    r2 = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 100, 'contractions': 0,
                   'spec_score': 0, 'voice_score': 0.0, 'hedges': 0},
        instr_density=None, word_count=100,
    )
    check("VSD ungated exactly 100 -> AMBER", r2.severity == 'AMBER', f"got {r2.severity}")


def test_vsd_ungated_yellow():
    """Ungated VSD in [21, 100) triggers YELLOW."""
    print("\n-- VSD UNGATED YELLOW --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis={'voice_gated': False, 'vsd': 30, 'contractions': 0,
                   'spec_score': 0, 'voice_score': 0.0, 'hedges': 0},
        instr_density=None,
        word_count=100,
    )
    check("VSD ungated 30 -> YELLOW", r.severity == 'YELLOW', f"got {r.severity}")
    check("VSD ungated 30 score == 0.30", r.score == 0.30, f"got {r.score}")


def test_no_signals_green():
    """All inputs below thresholds produce GREEN."""
    print("\n-- NO SIGNALS GREEN --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.05},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("No signals -> GREEN", r.severity == 'GREEN', f"got {r.severity}")
    check("No signals -> score 0.0", r.score == 0.0, f"got {r.score}")
    check("Explanation 'no signals'", 'no signals' in r.explanation)


def test_channel_name():
    """Channel result has correct channel name and mode_eligibility."""
    print("\n-- CHANNEL METADATA --")
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.0},
        voice_dis=_VD_NONE,
        instr_density=None,
        word_count=100,
    )
    check("channel name is prompt_structure", r.channel == 'prompt_structure')
    check("task_prompt in mode_eligibility", 'task_prompt' in r.mode_eligibility)
    check("generic_aigt in mode_eligibility", 'generic_aigt' in r.mode_eligibility)


if __name__ == '__main__':
    print("=" * 70)
    print("Prompt Structure Channel Tests")
    print("=" * 70)

    test_preamble_critical_instant_red()
    test_preamble_score_above_threshold()
    test_prompt_signature_red_threshold()
    test_prompt_signature_amber_threshold()
    test_prompt_signature_yellow_threshold()
    test_vsd_gated_red()
    test_vsd_gated_amber()
    test_idi_red()
    test_idi_amber()
    test_ssi_triggered_yellow()
    test_ssi_triggered_amber()
    test_ssi_not_triggered_word_count()
    test_ssi_not_triggered_high_voice_score()
    test_vsd_ungated_amber()
    test_vsd_ungated_yellow()
    test_no_signals_green()
    test_channel_name()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
