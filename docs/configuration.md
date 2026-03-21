# Configuration & Dependencies

> Feature flags, optional dependencies, project settings, and build configuration.

---

## Python Version

**Minimum:** Python 3.9+

The codebase avoids Python 3.10+ features (no walrus operator, no `match` statements) to maintain compatibility.

---

## Core Dependencies

These are always required:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 1.5 | Data loading and manipulation |
| openpyxl | ≥ 3.0 | Excel file reading |

---

## Optional Dependency Groups

Optional dependencies are organized into groups in `pyproject.toml`. Install only what you need:

### API Group (`pip install ".[api]"`)
| Package | Version | Purpose |
|---------|---------|---------|
| anthropic | ≥ 0.20 | DNA-GPT continuation via Anthropic API |
| openai | ≥ 1.0 | DNA-GPT continuation via OpenAI API |

### NLP Group (`pip install ".[nlp]"`)
| Package | Version | Purpose |
|---------|---------|---------|
| spacy | ≥ 3.0 | Accurate sentence tokenization |
| ftfy | ≥ 6.0 | Unicode encoding repair in normalization |
| sentence-transformers | ≥ 2.0 | Semantic embeddings (resonance, TOCSIN, flow) |
| scikit-learn | ≥ 1.0 | ML fusion, cosine similarity |

### Perplexity Group (`pip install ".[perplexity]"`)
| Package | Version | Purpose |
|---------|---------|---------|
| transformers | ≥ 4.20 | Language model loading |
| torch | ≥ 2.0 | Model inference for perplexity/binoculars |

### PDF Group (`pip install ".[pdf]"`)
| Package | Version | Purpose |
|---------|---------|---------|
| pypdf | ≥ 3.0 | PDF text extraction |

### Web Group (`pip install ".[web]"`)
| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.20 | Web dashboard interface |

### All Dependencies (`pip install ".[all]"`)
Installs all optional groups above.

### Bundle Group (`pip install ".[bundle]"`)
Installs all dependencies plus:
| Package | Version | Purpose |
|---------|---------|---------|
| pyinstaller | ≥ 6.0 | Standalone executable building |

---

## Feature Flags

Feature availability is managed in `llm_detector/compat.py`. Each flag is set at import time based on whether the dependency is available:

```python
from llm_detector.compat import (
    HAS_TK,          # tkinter → Desktop GUI
    HAS_SPACY,       # spaCy → Sentence tokenization
    HAS_FTFY,        # ftfy → Unicode repair
    HAS_SEMANTIC,    # sentence-transformers + sklearn → Embeddings
    HAS_PERPLEXITY,  # transformers + torch → Perplexity scoring
    HAS_BINOCULARS,  # Same as HAS_PERPLEXITY → Contrastive scoring
    HAS_PYPDF,       # pypdf → PDF loading
)
```

### What Each Flag Enables

| Flag | Analyzers Affected | Fallback Behavior |
|------|-------------------|-------------------|
| `HAS_SPACY` | All (sentence splitting) | Regex-based splitting |
| `HAS_FTFY` | Normalization | Skips encoding repair step |
| `HAS_SEMANTIC` | Semantic resonance, TOCSIN, Semantic flow, Similarity | Returns zero scores with explanatory reason |
| `HAS_PERPLEXITY` | Perplexity, Binoculars, Surprisal | Returns zero scores with explanatory reason |
| `HAS_PYPDF` | I/O loader (PDF) | PDF loading raises helpful error |
| `HAS_TK` | Desktop GUI | GUI entry point reports unavailable |

### Checking Feature Status

From the CLI:
```bash
llm-detector --precheck
```

From Python:
```python
from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
print(f"Semantic embeddings: {HAS_SEMANTIC}")
print(f"Perplexity models: {HAS_PERPLEXITY}")
```

---

## Detection Mode Configuration

| Mode | Description | Primary Channels |
|------|-------------|-----------------|
| `auto` | Auto-detects based on text characteristics | Depends on detection |
| `task_prompt` | Task prompts, evaluation items | Prompt Structure + Continuation |
| `generic_aigt` | Essays, reports, expository prose | All four channels |

Auto-detection heuristics:
- **task_prompt** if: prompt signature composite ≥ 0.15, OR instruction density IDI ≥ 5, OR framing completeness ≥ 2
- **generic_aigt** if: self-similarity signals ≥ 3, OR word count ≥ 400

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace API token for gated model access |
| `HUGGINGFACEHUB_API_TOKEN` | Alternative HuggingFace token variable |
| `ANTHROPIC_API_KEY` | Default Anthropic API key (can also be passed via `--api-key`) |
| `OPENAI_API_KEY` | Default OpenAI API key (can also be passed via `--api-key`) |

---

## Project Configuration (pyproject.toml)

Key configuration sections:

### Package Metadata
```toml
[project]
name = "llm-detector"
version = "0.68.0"
requires-python = ">=3.9"
license = "MIT"
```

### Entry Points
```toml
[project.scripts]
llm-detector = "llm_detector.cli:main"
llm-detector-dashboard = "llm_detector.cli:launch_dashboard"
llm-detector-gui = "llm_detector.cli:launch_gui"
```

### Test Configuration
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "requires_gpu: marks tests that require GPU",
    "requires_network: marks tests that require network"
]
```

### Coverage Configuration
```toml
[tool.coverage.run]
omit = [
    "llm_detector/dashboard.py",  # Streamlit (not unit-testable)
    "llm_detector/gui.py"         # Tkinter (not unit-testable)
]
```

---

## Installation Options

### Development Install
```bash
pip install -r requirements.txt
pip install pytest pytest-cov numpy pypdf
```

### Minimal Install (Core Only)
```bash
pip install pandas openpyxl
pip install -e .
```

### Full Install (All Features)
```bash
pip install -e ".[all]"
```

### Standalone Executable
See [INSTALL.md](../INSTALL.md) for PyInstaller build instructions.

---

## Directory Structure

| Directory/File | Purpose |
|---------------|---------|
| `llm_detector/` | Main Python package |
| `llm_detector/analyzers/` | 13+ detection analyzers |
| `llm_detector/channels/` | 4 scoring channels |
| `llm_detector/lexicon/` | 16 vocabulary packs |
| `tests/` | pytest test suite |
| `benchmarks/` | Fairness evaluation scripts |
| `.beet/` (runtime) | Memory store data directory |
| `dist/` (build) | PyInstaller output |

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Infrastructure: Feature Flags](infrastructure.md#compatibility--feature-flags) — Detailed flag reference
- [Testing Guide](testing.md) — Running tests
- [Installation Guide](../INSTALL.md) — Standalone executable builds
