# User Interfaces

> CLI, Desktop GUI (Tkinter), and Web Dashboard (Streamlit) — three ways to interact with the detection pipeline.

---

## Overview

The LLM Authorship Signal Analyzer provides three interfaces, all backed by the same core [pipeline](pipeline.md):

| Interface | Module | Entry Point | Best For |
|-----------|--------|-------------|----------|
| **CLI** | `llm_detector/cli.py` | `llm-detector` | Scripting, automation, batch processing |
| **Desktop GUI** | `llm_detector/gui.py` | `llm-detector-gui` | Interactive single-text analysis |
| **Web Dashboard** | `llm_detector/dashboard.py` | `llm-detector-dashboard` | Team-based batch review, visual reports |

---

## CLI (Command-Line Interface)

**Module:** `llm_detector/cli.py`
**Entry point:** `llm-detector` (or `python -m llm_detector`)

### Basic Usage

```bash
# Analyze a single text
llm-detector --text "Your prompt text here"

# Analyze from a file
llm-detector --file submissions.xlsx
llm-detector --file submissions.csv
llm-detector --file document.pdf

# Verbose output (full analyzer breakdown)
llm-detector --text "..." --verbose
```

### Column Mapping

When loading XLSX or CSV files, you can specify which columns contain the relevant data:

```bash
llm-detector --file data.xlsx \
    --prompt-col "Full Task Prompt" \
    --id-col "Task ID" \
    --occ-col "Occupation" \
    --attempter-col "Worker Name" \
    --stage-col "Pipeline Stage"
```

Columns accept names (matched case-insensitively) or Excel letter positions (A, B, C, ...).

### Detection Mode Control

```bash
# Force a specific detection mode
llm-detector --text "..." --mode task_prompt
llm-detector --text "..." --mode generic_aigt

# Auto-detect (default)
llm-detector --text "..." --mode auto
```

### Channel Ablation

```bash
# Disable specific channels
llm-detector --text "..." --disable-channel stylometric,continuation

# Run only prompt structure analysis
llm-detector --file data.xlsx --disable-channel stylometric,continuation,windowed
```

### DNA-GPT Continuation (API Mode)

```bash
# Using Anthropic API
llm-detector --text "..." --api-key sk-ant-... --dna-provider anthropic

# Using OpenAI API
llm-detector --text "..." --api-key sk-... --dna-provider openai
```

### Baseline Collection & Calibration

```bash
# Collect baselines from a batch
llm-detector --file data.xlsx --collect baselines.jsonl

# Analyze baselines
llm-detector --analyze-baselines baselines.jsonl

# Build calibration table
llm-detector --calibrate baselines.jsonl

# Use calibration for next batch
llm-detector --file data.xlsx --cal-table calibration.json
```

### Memory System (BEET)

```bash
# Initialize memory and process first batch
llm-detector --file batch1.xlsx --memory .beet/

# Process subsequent batches (compares against history)
llm-detector --file batch2.xlsx --memory .beet/

# Confirm labels from human review
llm-detector --memory .beet/ --confirm task_001 ai reviewer_name
llm-detector --memory .beet/ --confirm task_002 human reviewer_name

# View attempter history
llm-detector --memory .beet/ --attempter-history worker_42

# View memory summary
llm-detector --memory .beet/ --memory-summary

# Rebuild calibration from confirmed labels
llm-detector --memory .beet/ --rebuild-calibration
```

### Output Formats

```bash
# JSON output
llm-detector --file data.xlsx --output results.json

# CSV output
llm-detector --file data.xlsx --output results.csv

# HTML report
llm-detector --file data.xlsx --html-report report.html
```

### Quick Reference

| Flag | Description |
|------|-------------|
| `--text TEXT` | Analyze single text |
| `--file PATH` | Analyze file (XLSX/CSV/PDF) |
| `--verbose` | Full analyzer breakdown |
| `--mode MODE` | Force detection mode |
| `--disable-channel CH` | Disable channels (comma-separated) |
| `--api-key KEY` | API key for DNA-GPT |
| `--dna-provider NAME` | LLM provider (anthropic/openai) |
| `--collect PATH` | Collect baselines to JSONL |
| `--calibrate PATH` | Build calibration from baselines |
| `--cal-table PATH` | Apply calibration table |
| `--memory DIR` | Enable persistent memory |
| `--confirm ID LABEL REVIEWER` | Confirm a ground truth label |
| `--output PATH` | Write results to file |
| `--html-report PATH` | Generate HTML report |
| `--gui` | Open desktop GUI |
| `--version` | Show version |

---

## Desktop GUI (Tkinter)

**Module:** `llm_detector/gui.py`
**Entry point:** `llm-detector-gui` (or `llm-detector --gui`)
**Requires:** `HAS_TK` (tkinter)

### Overview

The desktop GUI provides an interactive interface with seven tabs:

| Tab | Purpose |
|-----|---------|
| **Analysis** | Single text or file analysis with real-time results |
| **Configuration** | Column mapping, mode selection, channel toggles |
| **Memory & Learning** | Memory store management, attempter history |
| **Calibration & Baselines** | Calibration table management, baseline analysis |
| **Reports** | Attempter profiling, financial impact, HTML reports |
| **Quick Reference** | Inline help and determination level guide |
| **Precheck** | Dependency status and system readiness check |

### Key Features

- **Drag-and-drop file loading** for XLSX, CSV, and PDF files
- **Real-time progress bar** during batch processing
- **Result export** to JSON, CSV, or HTML
- **Interactive detection span viewer** with color-coded highlights
- **Batch comparison mode** for side-by-side analysis

### Error Handling
All button handlers are wrapped in try/except blocks with `messagebox.showerror()` — the GUI never silently fails or crashes on exceptions.

---

## Web Dashboard (Streamlit)

**Module:** `llm_detector/dashboard.py`
**Entry point:** `llm-detector-dashboard`
**Requires:** Streamlit ≥ 1.20

### Overview

The web dashboard provides a browser-based interface for team-based batch review:

| Page | Purpose |
|------|---------|
| **Analysis** | Upload files and run batch analysis |
| **Configuration** | Column mapping, mode, channel settings |
| **Memory & Learning** | Memory store browsing, label confirmation |
| **Calibration & Baselines** | Calibration table management |
| **Reports** | Interactive charts, attempter profiles, financial impact |
| **Quick Reference** | Inline documentation |
| **Precheck** | System readiness and dependency check |

### Key Features

- **File upload** for XLSX, CSV, and PDF
- **Interactive filtering** by determination level, channel scores, and occupations
- **Severity distribution charts** with real-time updates
- **Confidence histogram** across the batch
- **Per-submission detail view** with detection spans
- **Report export** to HTML and CSV

### Launching

```bash
# Via entry point
llm-detector-dashboard

# Via CLI
llm-detector --dashboard

# Direct Streamlit command
streamlit run llm_detector/dashboard.py
```

### Security
The CLI and GUI validate the Streamlit dashboard module path via `os.path.realpath()` and require it to be within the `llm_detector` package directory before launching the subprocess. This prevents path traversal attacks.

---

## Interface Comparison

| Feature | CLI | GUI | Dashboard |
|---------|-----|-----|-----------|
| Single text analysis | ✓ | ✓ | ✓ |
| File batch processing | ✓ | ✓ | ✓ |
| Column mapping | ✓ | ✓ | ✓ |
| Mode selection | ✓ | ✓ | ✓ |
| Channel ablation | ✓ | ✓ | ✓ |
| DNA-GPT (API) | ✓ | ✓ | ✓ |
| Memory system | ✓ | ✓ | ✓ |
| Calibration | ✓ | ✓ | ✓ |
| HTML reports | ✓ | ✓ | ✓ |
| Interactive filtering | — | Limited | ✓ |
| Charts/visualization | — | — | ✓ |
| Scriptable | ✓ | — | — |
| Team sharing | — | — | ✓ |
| Progress bar | — | ✓ | ✓ |

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Pipeline Orchestration](pipeline.md) — What happens when you analyze text
- [Configuration & Dependencies](configuration.md) — Settings and requirements
