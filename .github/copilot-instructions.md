# Copilot Instructions

## Project Overview

**LLM Authorship Signal Analyzer for Human Data Pipelines** is a stylometric detection pipeline that identifies LLM-generated or LLM-assisted task prompts submitted through human data collection workflows. It is purpose-built for quality assurance in benchmark construction, clinical education assessments, and similar pipelines where human-authored content is expected.

The main Python package is `llm_detector` (entry point: `llm_detector/cli.py`). It ships three interfaces:
- **CLI** (`llm-detector`) — `llm_detector/cli.py`
- **Desktop GUI** (Tkinter) — `llm_detector/gui.py`
- **Web Dashboard** (Streamlit) — `llm_detector/dashboard.py`

## Repository Structure

```
llm_detector/
  analyzers/        # Individual detection layers (preamble, fingerprint, prompt_signature, …)
  channels/         # Scoring channels that aggregate analyzer outputs
  lexicon/          # Externalized vocabulary families (16 packs)
  cli.py            # CLI entry point and argument parsing
  gui.py            # Tkinter desktop GUI
  dashboard.py      # Streamlit web dashboard
  pipeline.py       # Core detection pipeline
  fusion.py         # Priority-based signal aggregation
  ml_fusion.py      # ML-based fusion layer
  io.py             # File I/O loaders (XLSX, CSV, TXT, PDF)
  memory.py         # Memory & learning subsystem
  calibration.py    # Calibration and baseline management
  similarity.py     # Pairwise similarity analysis
  reporting.py      # Report generation helpers
  html_report.py    # HTML report rendering
  compat.py         # Optional-dependency guards (HAS_SEMANTIC, HAS_PERPLEXITY, …)
tests/              # pytest test suite
```

## Build, Test, and Lint

**Install dependencies (development):**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov numpy pypdf
```

**Run the full test suite:**
```bash
python -m pytest -q
```

**Run tests with coverage (excluding untestable GUI modules):**
```bash
pytest -q --cov=llm_detector --cov-report=term-missing
```

Coverage measurement omits `llm_detector/dashboard.py` and `llm_detector/gui.py` (Streamlit/Tkinter GUIs are not unit-testable). This is configured in `pyproject.toml` under `[tool.coverage.run]`.

**No separate lint step is configured.** Keep code clean and consistent with existing style.

## Code Conventions

- **Python version**: 3.9+ compatible. No walrus operator, no `match` statements.
- **Imports**: Standard library first, then third-party, then local. Optional heavy dependencies (`torch`, `transformers`, `spacy`, `sentence-transformers`) are always guarded via `llm_detector/compat.py` flags (e.g., `HAS_SEMANTIC`, `HAS_PERPLEXITY`). Never unconditionally import these.
- **String style**: Use double quotes for strings.
- **Type hints**: Not required but welcome where they aid clarity.
- **Comments**: Only add comments where existing code has comments or when explaining non-obvious logic.
- **No new dependencies** should be added without updating both `requirements.txt` and `pyproject.toml`.

## Testing Practices

- All tests live in `tests/` and are discovered by pytest via `pyproject.toml`.
- Test files follow the naming convention `test_*.py`.
- Tests use a local `check(label, condition, detail="")` helper (defined per-file) instead of bare `assert` statements — follow this pattern when adding tests to existing files:
  ```python
  def check(label, condition, detail=""):
      global PASSED, FAILED
      if condition:
          PASSED += 1
          print(f"  [PASS] {label}")
      else:
          FAILED += 1
          print(f"  [FAIL] {label}  -- {detail}")
  ```
- Slow or network/GPU-requiring tests are marked with `@pytest.mark.slow`, `@pytest.mark.requires_network`, or `@pytest.mark.requires_gpu` and can be excluded with `-m "not slow"`.
- Optional-dependency tests must check the relevant `HAS_*` flag before exercising the feature.

## Key Patterns and Conventions

### Optional Dependency Guards
Always check `compat.py` flags before using heavy optional dependencies:
```python
from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
if HAS_SEMANTIC:
    # use sentence-transformers / sklearn
```

### Column Mapping
The pipeline supports 7 configurable columns: `task_id`, `occupation`, `attempter_name`, `pipeline_stage_name`, plus optional `attempter_email`, `reviewer`, and `reviewer_email`. These are passed as `id_col`, `occ_col`, `attempter_col`, `stage_col`, etc. to `load_xlsx` / `load_csv` in `io.py`. Column values accept either column names or Excel letters (A–Z).

### GUI Error Handling
All Tkinter button handlers must wrap logic in `try/except` blocks and display errors via `messagebox.showerror()` — never let exceptions silently fail in the GUI.

### Scoring and Fusion
Signals from individual analyzers flow through scoring channels (`channels/`) and are fused by `fusion.py` using priority-based aggregation and multi-layer convergence logic. The final label is one of `ai`, `human`, or `unsure`.

### Ground Truth Labels
The canonical label set is `['ai', 'human', 'unsure']` (lowercase). Use this set when validating or comparing labels.

## Entry Points

| Command | Description |
|---------|-------------|
| `llm-detector` | CLI (see `llm_detector/cli.py:main`) |
| `llm-detector-dashboard` | Streamlit web dashboard |
| `llm-detector-gui` | Tkinter desktop GUI |

Quick CLI usage:
```bash
llm-detector --text "Your prompt here"
llm-detector --file submissions.xlsx
```

## CI / GitHub Actions

The workflow in `.github/workflows/build.yml`:
1. Runs `pytest -q --cov=llm_detector --cov-report=xml --cov-report=term-missing` on Ubuntu with Python 3.11.
2. Builds a Windows standalone executable via PyInstaller.
3. Publishes a GitHub Release when a `v*` tag is pushed or `tag_name` is provided via `workflow_dispatch`.
