"""Tests for the full pipeline integration."""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.pipeline import analyze_prompt
from llm_detector.compat import HAS_PYPDF, HAS_SEMANTIC, HAS_PERPLEXITY, HAS_FTFY
from llm_detector.io import load_pdf
from tests.conftest import CLINICAL_TEXT

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


def test_pdf_loading():
    print("\n-- PDF LOADING --")

    if HAS_PYPDF:
        tmpdir = tempfile.mkdtemp()
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            pdf_path = os.path.join(tmpdir, 'test.pdf')
            with open(pdf_path, 'wb') as f:
                writer.write(f)

            tasks = load_pdf(pdf_path)
            check("PDF loads without error", isinstance(tasks, list))
            check("Blank PDF: 0 tasks", len(tasks) == 0,
                  f"got {len(tasks)} tasks")
        finally:
            shutil.rmtree(tmpdir)
    else:
        print("  (pypdf not installed -- skipping PDF tests)")
        tasks = load_pdf("/nonexistent.pdf")
        check("No pypdf: returns empty list", tasks == [])


def test_pipeline_v066():
    print("\n-- FULL PIPELINE v0.66 INTEGRATION --")

    text = CLINICAL_TEXT * 3
    r = analyze_prompt(text, task_id='v066_test', run_l3=True, mode='auto')

    # v0.61 fields present (using new descriptive names)
    check("semantic_resonance_ai_score in result", 'semantic_resonance_ai_score' in r)
    check("semantic_resonance_delta in result", 'semantic_resonance_delta' in r)
    check("semantic_resonance_determination in result", 'semantic_resonance_determination' in r)
    check("perplexity_value in result", 'perplexity_value' in r)
    check("perplexity_determination in result", 'perplexity_determination' in r)
    check("surprisal_variance in result", 'surprisal_variance' in r)
    check("volatility_decay_ratio in result", 'volatility_decay_ratio' in r)

    # Audit trail
    at = r.get('audit_trail', {})
    check("audit_trail version is v0.66", at.get('pipeline_version') == 'v0.66',
          f"got {at.get('pipeline_version')}")
    check("audit_trail has semantic_available", 'semantic_available' in at)
    check("audit_trail has perplexity_available", 'perplexity_available' in at)
    check("audit_trail norm has ftfy_applied", 'ftfy_applied' in at.get('normalization', {}))

    # Core fields present
    check("norm fields", 'norm_obfuscation_delta' in r)
    check("fairness fields", 'lang_support_level' in r)
    check("mode field", 'mode' in r)
    check("channel_details", 'channel_details' in r)
    check("window fields", 'window_max_score' in r)
    check("stylo fields", 'stylo_fw_ratio' in r)
    check("calibrated_confidence", 'calibrated_confidence' in r)
    check("conformity_level", 'conformity_level' in r)
    check("self_similarity fields", 'self_similarity_nssi_score' in r)

    # v0.65 new fields
    check("continuation_composite_stability", 'continuation_composite_stability' in r)
    check("continuation_improvement_rate", 'continuation_improvement_rate' in r)
    check("continuation_ncd_matrix_variance", 'continuation_ncd_matrix_variance' in r)
    check("window_fw_trajectory_cv", 'window_fw_trajectory_cv' in r)
    check("window_comp_trajectory_cv", 'window_comp_trajectory_cv' in r)
    check("tocsin_cohesiveness", 'tocsin_cohesiveness' in r)
    check("perplexity_zlib_normalized_ppl", 'perplexity_zlib_normalized_ppl' in r)
    check("self_similarity_structural_compression_delta", 'self_similarity_structural_compression_delta' in r)
    check("surprisal_trajectory_cv", 'surprisal_trajectory_cv' in r)

    # Channel structure
    cd = r.get('channel_details', {})
    check("4 channels", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")
    for ch in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
        check(f"Channel {ch} present", ch in cd.get('channels', {}))


def test_pipeline_with_local_proxy():
    print("\n-- FULL PIPELINE WITH LOCAL PROXY --")

    text = CLINICAL_TEXT * 3

    r = analyze_prompt(text, task_id='proxy_test', run_l3=True, mode='auto')

    check("continuation_mode is 'local'", r.get('continuation_mode') == 'local',
          f"got {r.get('continuation_mode')}")
    check("continuation_ncd in result", 'continuation_ncd' in r)
    check("continuation_internal_overlap in result", 'continuation_internal_overlap' in r)
    check("continuation_composite in result", 'continuation_composite' in r)
    check("continuation_ttr in result", 'continuation_ttr' in r)
    check("continuation_cond_surprisal in result", 'continuation_cond_surprisal' in r)
    check("continuation_repeat4 in result", 'continuation_repeat4' in r)

    ncd = r.get('continuation_ncd', 0)
    check("NCD in plausible range", 0.0 <= ncd <= 1.2,
          f"got {ncd}")

    cd = r.get('channel_details', {})
    cont_ch = cd.get('channels', {}).get('continuation', {})
    check("Continuation channel has score", 'score' in cont_ch)


if __name__ == '__main__':
    print("=" * 70)
    print("Pipeline Integration Tests")
    print("=" * 70)

    test_pdf_loading()
    test_pipeline_v066()
    test_pipeline_with_local_proxy()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
