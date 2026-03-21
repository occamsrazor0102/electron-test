"""
LLM-Generated Task Prompt Detection Pipeline
===================================================
Multi-layer stylometric detection pipeline for identifying LLM-generated
or LLM-assisted task prompts in human data collection workflows.

Package structure:
    analyzers/     - Individual detection layers (preamble, fingerprint, etc.)
    channels/      - Channel scoring (prompt_structure, stylometric, etc.)
    lexicon/       - Externalized versioned detection vocabulary
    fusion.py      - Evidence fusion across channels
    pipeline.py    - Full pipeline orchestration
    calibration.py - Conformal calibration
    baselines.py   - Baseline collection and analysis
    similarity.py  - Cross-submission similarity
    io.py          - File I/O (XLSX, CSV, PDF)
    cli.py         - Command-line interface
    gui.py         - Desktop GUI
"""

__version__ = '0.68.0'

# Core pipeline
from llm_detector.pipeline import analyze_prompt
from llm_detector.normalize import normalize_text
from llm_detector.language_gate import check_language_support
from llm_detector.fusion import determine
from llm_detector.channels import ChannelResult

# Feature flags
from llm_detector.compat import (
    HAS_SPACY, HAS_FTFY, HAS_SEMANTIC, HAS_PERPLEXITY, HAS_PYPDF, HAS_TK,
)

# Analyzers
from llm_detector.analyzers import (
    run_preamble,
    run_fingerprint,
    run_prompt_signature,
    run_voice_dissonance,
    run_instruction_density,
    run_semantic_resonance,
    run_self_similarity,
    run_continuation_api,
    run_continuation_api_multi,
    run_continuation_local,
    run_perplexity,
)

# Infrastructure
from llm_detector.calibration import (
    calibrate_from_baselines, apply_calibration,
    save_calibration, load_calibration,
)
from llm_detector.baselines import collect_baselines, analyze_baselines
from llm_detector.similarity import analyze_similarity, print_similarity_report
from llm_detector.io import load_xlsx, load_csv, load_pdf
