# Testing Guide

> Test conventions, running tests, coverage configuration, and how to add new tests.

---

## Overview

The project has a comprehensive test suite with 33 test files covering all major components. Tests use pytest and follow specific conventions documented below.

---

## Running Tests

### Full Test Suite
```bash
python -m pytest -q
```

### With Coverage
```bash
pytest -q --cov=llm_detector --cov-report=term-missing
```

### With XML Coverage Report (CI)
```bash
pytest -q --cov=llm_detector --cov-report=xml --cov-report=term-missing
```

### Excluding Slow Tests
```bash
pytest -q -m "not slow"
```

### Running a Single Test File
```bash
pytest tests/test_analyzers.py -v
```

### Running a Specific Test
```bash
pytest tests/test_analyzers.py::test_preamble_basic -v
```

---

## Test Conventions

### File Naming
All test files live in `tests/` and follow the naming convention `test_*.py`.

### The `check()` Helper Pattern

Tests use a local `check(label, condition, detail="")` helper instead of bare `assert` statements. This provides verbose pass/fail output during test runs:

```python
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
```

When adding tests to existing files, follow this pattern. Each test function defines its own `check()` helper with module-level `PASSED`/`FAILED` counters, and uses `assert FAILED == 0` at the end to fail the test if any check failed.

### Test Markers

| Marker | Purpose | Usage |
|--------|---------|-------|
| `@pytest.mark.slow` | Tests that take >5 seconds | `pytest -m "not slow"` to skip |
| `@pytest.mark.requires_gpu` | Tests requiring GPU hardware | Skipped in CI by default |
| `@pytest.mark.requires_network` | Tests requiring internet access | Skipped in offline environments |

### Optional Dependency Testing

Tests that exercise optional-dependency features must check the relevant `HAS_*` flag:

```python
from llm_detector.compat import HAS_SEMANTIC

def test_semantic_resonance():
    if not HAS_SEMANTIC:
        pytest.skip("sentence-transformers not available")
    # ... test code ...
```

---

## Test File Index

| Test File | Module(s) Tested | Description |
|-----------|-----------------|-------------|
| `test_analyzers.py` | `analyzers/*` | Individual analyzer unit tests (preamble, fingerprint, etc.) |
| `test_preamble_cot.py` | `analyzers/preamble.py` | Chain-of-Thought leakage detection (`<think>` tags) |
| `test_prompt_structure_channel.py` | `channels/prompt_structure.py` | Prompt structure channel scoring |
| `test_channels_prompt_structure.py` | `channels/prompt_structure.py` | Extended prompt structure channel tests |
| `test_stylometric_channel.py` | `channels/stylometric.py` | NSSI, perplexity, semantic resonance channel |
| `test_channels_stylometric.py` | `channels/stylometric.py` | Extended stylometric channel tests |
| `test_continuation_local.py` | `analyzers/continuation_local.py` | DNA-GPT local proxy + NCD matrix |
| `test_dna_truncation.py` | `analyzers/continuation_api.py` | Multi-truncation stability (γ=0.3/0.5/0.7) |
| `test_semantic_flow.py` | `analyzers/semantic_flow.py` | Inter-sentence variance detection |
| `test_token_cohesiveness.py` | `analyzers/token_cohesiveness.py` | TOCSIN deletion stability |
| `test_windowed.py` | `analyzers/windowing.py` | Window scoring, changepoint, mixed content |
| `test_fusion.py` | `fusion.py` | Channel aggregation, mode detection |
| `test_fusion_edge_cases.py` | `fusion.py` | Boundary cases, empty text, short text |
| `test_calibration.py` | `calibration.py` | Conformal calibration from baselines |
| `test_baselines_collection.py` | `baselines.py` | JSONL accumulation, statistics |
| `test_lexicon.py` | `lexicon/*` | Lexicon pack scoring |
| `test_similarity.py` | `similarity.py` | Cross-submission matching |
| `test_normalize.py` | `normalize.py` | Obfuscation detection (homoglyphs, zero-width) |
| `test_pipeline.py` | `pipeline.py` | Full-pipeline integration |
| `test_cli.py` | `cli.py` | Command-line argument parsing + batch mode |
| `test_io_loaders.py` | `io.py` | XLSX/CSV/PDF loading |
| `test_html_report.py` | `html_report.py` | HTML generation |
| `test_reporting.py` | `reporting.py` | Result serialization |
| `test_ml_fusion.py` | `ml_fusion.py` | ML fusion (LogisticRegression, RandomForest) |
| `test_ml_mocked.py` | `ml_fusion.py` | ML fusion with mocked training |
| `test_gui_dashboard_features.py` | `gui.py`, `dashboard.py` | GUI/Dashboard Streamlit interaction |
| `test_memory.py` | `memory.py` | Runtime memory management |
| `test_mattr_burstiness.py` | `analyzers/self_similarity.py` | Moving-average TTR + burstiness |
| `test_xray_spans.py` | `pipeline.py` | Detection span extraction & highlighting |
| `test_hf_token.py` | `compat.py` | HuggingFace token handling |

---

## Coverage Configuration

Coverage measurement is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
omit = [
    "llm_detector/dashboard.py",  # Streamlit (not unit-testable)
    "llm_detector/gui.py"         # Tkinter (not unit-testable)
]
```

**Why these are excluded:**
- `dashboard.py` — Streamlit applications cannot be meaningfully unit-tested
- `gui.py` — Tkinter GUI requires display server and user interaction

**Minimum coverage target:** 60% (excluding GUI modules)

---

## CI / GitHub Actions

The CI workflow (`.github/workflows/build.yml`) runs on every push and PR:

1. **Test step:** `pytest -q --cov=llm_detector --cov-report=xml --cov-report=term-missing` on Ubuntu with Python 3.11
2. **Build step:** Builds a Windows standalone executable via PyInstaller
3. **Release step:** Publishes a GitHub Release when a `v*` tag is pushed

---

## Adding New Tests

### Step 1: Create a test file

```python
# tests/test_my_feature.py
import pytest

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

def test_my_feature():
    global PASSED, FAILED
    PASSED = FAILED = 0

    # ... setup ...

    check("basic functionality", result == expected,
          f"got {result}")

    check("edge case", edge_result is not None)

    assert FAILED == 0, f"{FAILED} checks failed"
```

### Step 2: Use optional dependency guards

```python
from llm_detector.compat import HAS_SEMANTIC

def test_semantic_feature():
    if not HAS_SEMANTIC:
        pytest.skip("requires sentence-transformers")
    # ... test code ...
```

### Step 3: Mark slow tests

```python
@pytest.mark.slow
def test_expensive_operation():
    # ... test that takes >5 seconds ...
```

### Step 4: Run and verify

```bash
# Run just your new test
pytest tests/test_my_feature.py -v

# Run full suite to verify no regressions
python -m pytest -q
```

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Configuration & Dependencies](configuration.md) — Project settings
