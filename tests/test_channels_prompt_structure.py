"""Comprehensive tests for channels/prompt_structure.py.

Tests focus on critical threshold boundaries and edge cases identified in coverage analysis.
"""

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


def test_preamble_critical_path():
    """Test preamble CRITICAL severity triggers immediate RED."""
    print("\n-- PREAMBLE CRITICAL PATH --")

    # CRITICAL preamble should immediately return RED with 0.99 score
    result = score_prompt_structure(
        preamble_score=0.80,
        preamble_severity='CRITICAL',
        prompt_sig={'composite': 0.10},
        voice_dis={'voice_gated': False, 'vsd': 0, 'voice_score': 0, 'spec_score': 0,
                   'contractions': 5, 'hedges': 3},
        instr_density={'idi': 2.0},
        word_count=100
    )
    check("CRITICAL preamble -> RED determination", result.severity == 'RED')
    check("CRITICAL preamble -> 0.99 score", result.score == 0.99)
    check("CRITICAL preamble explanation", 'Preamble detection (critical hit)' in result.explanation)


def test_prompt_signature_thresholds():
    """Test all three prompt signature thresholds (0.20, 0.40, 0.60)."""
    print("\n-- PROMPT SIGNATURE THRESHOLDS --")

    # Base inputs for non-prompt-sig signals
    voice_dis = {'voice_gated': False, 'vsd': 0, 'voice_score': 0, 'spec_score': 0,
                 'contractions': 5, 'hedges': 3}
    instr_density = None

    # RED threshold: composite >= 0.60
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.65},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig 0.65 -> RED", r.severity == 'RED')
    check("prompt_sig 0.65 -> score == composite", r.score == 0.65)

    # AMBER threshold: 0.40 <= composite < 0.60
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.45},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig 0.45 -> AMBER", r.severity == 'AMBER')
    check("prompt_sig 0.45 -> score == composite", r.score == 0.45)

    # YELLOW threshold: 0.20 <= composite < 0.40
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.25},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig 0.25 -> YELLOW", r.severity == 'YELLOW')
    check("prompt_sig 0.25 -> score reduced by 0.7 factor", abs(r.score - 0.25 * 0.7) < 0.01)

    # Below threshold: composite < 0.20
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.15},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig 0.15 -> GREEN (below threshold)", r.severity == 'GREEN')

    # Boundary tests
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.60},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig exactly 0.60 -> RED", r.severity == 'RED')

    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.40},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig exactly 0.40 -> AMBER", r.severity == 'AMBER')

    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.20},
        voice_dis=voice_dis,
        instr_density=instr_density,
        word_count=100
    )
    check("prompt_sig exactly 0.20 -> YELLOW", r.severity == 'YELLOW')


def test_vsd_gated_thresholds():
    """Test voice dissonance gated path with 21 and 50 thresholds."""
    print("\n-- VSD GATED THRESHOLDS --")

    prompt_sig = {'composite': 0.10}

    # VSD >= 50 (gated) -> RED
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': True, 'vsd': 55, 'voice_score': 0, 'spec_score': 10,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("VSD gated >= 50 -> RED", r.severity == 'RED')
    check("VSD gated >= 50 -> score 0.90", r.score == 0.90)
    check("VSD in sub_signals", 'vsd_gated' in r.sub_signals)

    # VSD >= 21 (gated) -> AMBER
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': True, 'vsd': 30, 'voice_score': 0, 'spec_score': 8,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("VSD gated >= 21 -> AMBER", r.severity == 'AMBER')
    check("VSD gated >= 21 -> score 0.70", r.score == 0.70)

    # VSD < 21 (gated) -> no signal
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': True, 'vsd': 15, 'voice_score': 0.1, 'spec_score': 3,
                   'contractions': 1, 'hedges': 1},
        instr_density=None,
        word_count=200
    )
    check("VSD gated < 21 -> GREEN", r.severity == 'GREEN')

    # Boundary test: VSD exactly 50
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': True, 'vsd': 50, 'voice_score': 0, 'spec_score': 10,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("VSD gated exactly 50 -> RED", r.severity == 'RED')

    # Boundary test: VSD exactly 21
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': True, 'vsd': 21, 'voice_score': 0, 'spec_score': 7,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("VSD gated exactly 21 -> AMBER", r.severity == 'AMBER')


def test_vsd_ungated_thresholds():
    """Test voice dissonance ungated path with 100 and 21 thresholds."""
    print("\n-- VSD UNGATED THRESHOLDS --")

    prompt_sig = {'composite': 0.10}

    # VSD >= 100 (ungated) -> AMBER
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 120, 'voice_score': 0.3, 'spec_score': 15,
                   'contractions': 2, 'hedges': 1},
        instr_density=None,
        word_count=200
    )
    check("VSD ungated >= 100 -> AMBER", r.severity == 'AMBER')
    check("VSD ungated >= 100 -> score 0.60", r.score == 0.60)
    check("VSD ungated in sub_signals", 'vsd_ungated' in r.sub_signals)

    # VSD >= 21 but < 100 (ungated) -> YELLOW
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 50, 'voice_score': 0.4, 'spec_score': 8,
                   'contractions': 3, 'hedges': 2},
        instr_density=None,
        word_count=200
    )
    check("VSD ungated >= 21, < 100 -> YELLOW", r.severity == 'YELLOW')
    check("VSD ungated >= 21, < 100 -> score 0.30", r.score == 0.30)

    # VSD < 21 (ungated) -> no signal
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 15, 'voice_score': 0.5, 'spec_score': 3,
                   'contractions': 5, 'hedges': 3},
        instr_density=None,
        word_count=200
    )
    check("VSD ungated < 21 -> GREEN", r.severity == 'GREEN')

    # Boundary tests
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 100, 'voice_score': 0.3, 'spec_score': 12,
                   'contractions': 2, 'hedges': 1},
        instr_density=None,
        word_count=200
    )
    check("VSD ungated exactly 100 -> AMBER", r.severity == 'AMBER')


def test_instruction_density_thresholds():
    """Test instruction density IDI thresholds (8 and 12)."""
    print("\n-- INSTRUCTION DENSITY THRESHOLDS --")

    prompt_sig = {'composite': 0.10}
    voice_dis = {'voice_gated': False, 'vsd': 0, 'voice_score': 0.5, 'spec_score': 2,
                 'contractions': 5, 'hedges': 3}

    # IDI >= 12 -> RED
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis=voice_dis,
        instr_density={'idi': 15},
        word_count=200
    )
    check("IDI >= 12 -> RED", r.severity == 'RED')
    check("IDI >= 12 -> score 0.85", r.score == 0.85)
    check("IDI in sub_signals", 'idi' in r.sub_signals)

    # IDI >= 8 but < 12 -> AMBER
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis=voice_dis,
        instr_density={'idi': 10},
        word_count=200
    )
    check("IDI >= 8, < 12 -> AMBER", r.severity == 'AMBER')
    check("IDI >= 8, < 12 -> score 0.65", r.score == 0.65)

    # IDI < 8 -> no signal
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis=voice_dis,
        instr_density={'idi': 5},
        word_count=200
    )
    check("IDI < 8 -> GREEN", r.severity == 'GREEN')

    # Boundary tests
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis=voice_dis,
        instr_density={'idi': 12},
        word_count=200
    )
    check("IDI exactly 12 -> RED", r.severity == 'RED')

    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis=voice_dis,
        instr_density={'idi': 8},
        word_count=200
    )
    check("IDI exactly 8 -> AMBER", r.severity == 'AMBER')


def test_ssi_triggering_logic():
    """Test SSI (Sterile Specification Index) triggering conditions."""
    print("\n-- SSI TRIGGERING LOGIC --")

    prompt_sig = {'composite': 0.10}

    # SSI triggered with high spec_score (>= 8.0), no contractions -> AMBER
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.3, 'spec_score': 9.0,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("SSI high spec >= 8.0 -> AMBER", r.severity == 'AMBER')
    check("SSI high -> score 0.70", r.score == 0.70)
    check("SSI in sub_signals", 'ssi' in r.sub_signals)

    # SSI triggered with medium spec_score (>= 5.0, < 8.0), no contractions -> YELLOW
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.2, 'spec_score': 6.0,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("SSI medium spec >= 5.0, < 8.0 -> YELLOW", r.severity == 'YELLOW')
    check("SSI medium -> score 0.45", r.score == 0.45)

    # SSI with contractions -> higher threshold (7.0 instead of 5.0)
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.2, 'spec_score': 6.0,
                   'contractions': 2, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("SSI with contractions: spec 6.0 < threshold 7.0 -> GREEN", r.severity == 'GREEN')

    # SSI with contractions and spec >= 7.0 -> should trigger
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.2, 'spec_score': 7.5,
                   'contractions': 2, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("SSI with contractions: spec >= 7.0 -> YELLOW", r.severity == 'YELLOW')

    # SSI fails if voice_score >= 0.5
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.6, 'spec_score': 9.0,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=200
    )
    check("SSI fails if voice_score >= 0.5 -> GREEN", r.severity == 'GREEN')

    # SSI fails if hedges > 0
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.2, 'spec_score': 9.0,
                   'contractions': 0, 'hedges': 2},
        instr_density=None,
        word_count=200
    )
    check("SSI fails if hedges > 0 -> GREEN", r.severity == 'GREEN')

    # SSI fails if word_count < 150
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.2, 'spec_score': 9.0,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=100
    )
    check("SSI fails if word_count < 150 -> GREEN", r.severity == 'GREEN')

    # SSI at boundary word_count = 150
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig=prompt_sig,
        voice_dis={'voice_gated': False, 'vsd': 10, 'voice_score': 0.2, 'spec_score': 9.0,
                   'contractions': 0, 'hedges': 0},
        instr_density=None,
        word_count=150
    )
    check("SSI at word_count = 150 -> AMBER", r.severity == 'AMBER')


def test_multiple_signals_escalation():
    """Test that multiple signals properly escalate severity."""
    print("\n-- MULTIPLE SIGNALS ESCALATION --")

    # YELLOW prompt_sig + YELLOW IDI -> should escalate to higher severity
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.25},
        voice_dis={'voice_gated': False, 'vsd': 0, 'voice_score': 0.5, 'spec_score': 2,
                   'contractions': 5, 'hedges': 3},
        instr_density={'idi': 9},
        word_count=200
    )
    check("YELLOW prompt_sig + AMBER IDI -> escalates to AMBER", r.severity == 'AMBER')

    # Multiple RED signals
    r = score_prompt_structure(
        preamble_score=0, preamble_severity='GREEN',
        prompt_sig={'composite': 0.65},
        voice_dis={'voice_gated': True, 'vsd': 55, 'voice_score': 0, 'spec_score': 10,
                   'contractions': 0, 'hedges': 0},
        instr_density={'idi': 15},
        word_count=200
    )
    check("Multiple RED signals -> RED", r.severity == 'RED')
    check("Multiple RED signals -> max score taken", r.score == 0.90)


def test_preamble_non_critical():
    """Test that non-CRITICAL preamble still contributes to scoring."""
    print("\n-- PREAMBLE NON-CRITICAL --")

    # Preamble score >= 0.50 should contribute
    r = score_prompt_structure(
        preamble_score=0.70,
        preamble_severity='AMBER',
        prompt_sig={'composite': 0.10},
        voice_dis={'voice_gated': False, 'vsd': 0, 'voice_score': 0.5, 'spec_score': 2,
                   'contractions': 5, 'hedges': 3},
        instr_density=None,
        word_count=100
    )
    check("Preamble >= 0.50 contributes to score", r.score >= 0.70)
    check("Preamble in sub_signals", 'preamble' in r.sub_signals)

    # Preamble score < 0.50 should not contribute
    r = score_prompt_structure(
        preamble_score=0.30,
        preamble_severity='YELLOW',
        prompt_sig={'composite': 0.10},
        voice_dis={'voice_gated': False, 'vsd': 0, 'voice_score': 0.5, 'spec_score': 2,
                   'contractions': 5, 'hedges': 3},
        instr_density=None,
        word_count=100
    )
    check("Preamble < 0.50 does not contribute", 'preamble' not in r.sub_signals)


def test_no_signals_baseline():
    """Test that no signals produce GREEN with zero score."""
    print("\n-- NO SIGNALS BASELINE --")

    r = score_prompt_structure(
        preamble_score=0,
        preamble_severity='GREEN',
        prompt_sig={'composite': 0.05},
        voice_dis={'voice_gated': False, 'vsd': 5, 'voice_score': 0.8, 'spec_score': 1,
                   'contractions': 10, 'hedges': 5},
        instr_density={'idi': 2},
        word_count=100
    )
    check("No signals -> GREEN", r.severity == 'GREEN')
    check("No signals -> score near 0", r.score < 0.1)


if __name__ == '__main__':
    test_preamble_critical_path()
    test_prompt_signature_thresholds()
    test_vsd_gated_thresholds()
    test_vsd_ungated_thresholds()
    test_instruction_density_thresholds()
    test_ssi_triggering_logic()
    test_multiple_signals_escalation()
    test_preamble_non_critical()
    test_no_signals_baseline()

    print(f"\n{'='*60}")
    print(f"TOTAL: {PASSED} passed, {FAILED} failed")
    print(f"{'='*60}")

    if FAILED > 0:
        sys.exit(1)
