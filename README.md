# LLM Authorship Signal Analyzer for Human Data Pipelines

[![Latest Release](https://img.shields.io/github/v/release/occamsrazor0102/-LLM-Authorship-Signal-Analyzer-for-Human-Data-Pipelines?label=latest%20release)](https://github.com/occamsrazor0102/-LLM-Authorship-Signal-Analyzer-for-Human-Data-Pipelines/releases/latest)

A stylometric detection pipeline that identifies LLM-generated or LLM-assisted task prompts submitted through human data collection workflows.

## Download (Standalone Executables)

Pre-built single-file executables — no Python or dependencies required. Download the file for your platform from the [**latest release**](https://github.com/occamsrazor0102/-LLM-Authorship-Signal-Analyzer-for-Human-Data-Pipelines/releases/latest):

| Platform | File |
|----------|------|
| **Windows** | `llm-detector-windows-x86_64.exe` |
| **macOS (Intel)** | `llm-detector-macos-x86_64` |
| **macOS (Apple Silicon)** | `llm-detector-macos-arm64` |
| **Linux** | `llm-detector-linux-x86_64` |

**macOS / Linux — make executable after download:**
```bash
chmod +x llm-detector-macos-arm64   # or the variant you downloaded
./llm-detector-macos-arm64 --text "Your prompt here"
```

**Windows — run from PowerShell or Command Prompt:**
```
.\llm-detector-windows-x86_64.exe --text "Your prompt here"
```

Designed for quality assurance in benchmark construction (GDPval-style evaluation tasks), clinical education assessments, and any pipeline where humans are expected to author original prompts but may submit LLM-generated content instead.

### Monolith Repo Only

- Chain-of-thought leakage detection (`<think>` tags, reasoning-model phrases)
- DivEye-inspired surprisal variance and volatility decay in perplexity analysis
- Multi-truncation stability and cross-prefix surprisal curves (v0.65)
- Token cohesiveness (TOCSIN) and CUSUM changepoint detection (v0.65)
- Simpler file structure optimized for single-file distribution

## The Problem

Human data pipelines — where workers author task prompts, evaluation scenarios, or assessment items — are vulnerable to a specific failure mode: a contributor uses an LLM to generate their submission rather than writing it themselves. This degrades data quality because LLM-generated prompts exhibit systematic biases in structure, vocabulary, and specification patterns that contaminate the resulting benchmark.

Standard AI-text detectors (GPTZero, Originality.ai, etc.) are trained on prose and perform poorly on task prompts, which are inherently instructional and specification-heavy. This tool is purpose-built for that domain.

## How It Works

The pipeline analyzes text across multiple independent layers, each targeting a different authorship signal. No single layer is definitive — the system combines evidence across layers using priority-based aggregation and multi-layer convergence logic.

### Detection Layers

| Layer | Module | Description |
|-------|--------|-------------|
| **Preamble** | `analyzers/preamble.py` | Catches LLM output artifacts: assistant acknowledgments, artifact delivery frames, first-person creation claims, meta-design language |
| **Fingerprint** | `analyzers/fingerprint.py` | 27-word diagnostic vocabulary supplemented by 16 typed lexicon pack families (diagnostic only, not standalone trigger) |
| **Prompt Signature** | `analyzers/prompt_signature.py` | Structural patterns of LLM-generated prompts: Constraint Frame Density, Must-Frame Saturation Rate, meta-evaluation design language |
| **Voice Dissonance** | `analyzers/voice_dissonance.py` | Measures contradiction between casual voice markers and technical specification density |
| **Instruction Density** | `analyzers/instruction_density.py` | Counts formal-exhaustive specification patterns: imperatives, conditionals, binary specs |
| **Semantic Resonance** | `analyzers/semantic_resonance.py` | Cosine similarity of sentence embeddings against AI/human archetype centroids |
| **Self-Similarity** | `analyzers/self_similarity.py` | N-gram Self-Similarity Index (NSSI) for detecting formulaic LLM patterns |
| **Continuation (API)** | `analyzers/continuation_api.py` | DNA-GPT divergent continuation analysis via Anthropic/OpenAI API |
| **Continuation (Local)** | `analyzers/continuation_local.py` | Zero-LLM DNA-GPT proxy using backoff n-gram language model |
| **Perplexity** | `analyzers/perplexity.py` | distilgpt2-based perplexity scoring |
| **Token Cohesiveness** | `analyzers/token_cohesiveness.py` | TOCSIN: semantic fragility under random word deletion |
| **Windowing** | `analyzers/windowing.py` | Sentence-window analysis with FW trajectory, compression profile, and CUSUM changepoint detection |
| **Semantic Flow** | `analyzers/semantic_flow.py` | Inter-sentence embedding similarity variance — LLMs produce uniformly smooth transitions (low variance); human writing jumps erratically (high variance) |
| **Stylometry** | `analyzers/stylometry.py` | Topic-scrubbed stylometric features: MATTR, function-word ratio, sentence-length dispersion, lexical richness — computed after masking topical content to reduce domain leakage |

### Scoring Channels

Signals are organized into four independent scoring channels:

| Channel | Module | Primary Layers |
|---------|--------|----------------|
| **Prompt Structure** | `channels/prompt_structure.py` | Preamble, Prompt Signature, Voice Dissonance, Instruction Density |
| **Stylometric** | `channels/stylometric.py` | Self-Similarity, Semantic Resonance, Perplexity, Fingerprint, TOCSIN, Semantic Flow |
| **Continuation** | `channels/continuation.py` | Continuation API or Local (multi-truncation, NCD matrix) |
| **Windowed** | `channels/windowed.py` | Sentence-window scoring (FW trajectory, compression profile, changepoint) |

### Lexicon Pack System

The pipeline includes 16 externalized vocabulary families organized by semantic category, each with independent weights and caps. Packs feed into specific layers:

| Category | Packs | Target Layer |
|----------|-------|-------------|
| Constraint | obligation, prohibition, recommendation, conditional, cardinality, state | Prompt Signature |
| Schema | schema_json, schema_types, data_fields, tabular | Voice Dissonance |
| Exec-Spec | gherkin, rubric, acceptance | Prompt Signature |
| Instruction | task_verbs, value_domain | Instruction Density |
| Format | format_markup | Voice Dissonance |

### v0.65 Detection Signals

Ten new signals added in v0.65, exploiting temporal uniformity and compressibility patterns:

| Signal | Channel | Description |
|--------|---------|-------------|
| Multi-truncation stability | Continuation | Composite score variance across γ=0.3/0.5/0.7 truncation points |
| Cross-prefix surprisal curve | Continuation | How predictability improves with more context (human: 15–40%, AI: 0–10%) |
| Multi-segment NCD matrix | Continuation | Pairwise compression distance across 4 text segments |
| Function word trajectory | Windowed | CV of function word ratio across sentence windows |
| Windowed compression profile | Windowed | Per-window zlib compression ratio uniformity |
| CUSUM changepoint | Windowed | Detects human→AI transition boundaries via effect size |
| Surprisal trajectory | Windowed | Windowed token-level surprisal statistics (requires `transformers`) |
| Structural compression delta (s13) | Self-Similarity | Original vs word-shuffled compression ratio |
| Zlib-normalized perplexity | Perplexity | PPL × compression ratio compound signal |
| Token cohesiveness (TOCSIN) | Stylometric | Semantic fragility under random word deletion (requires `sentence-transformers`) |

### Detection Modes

| Mode | Primary Channels | Use Case |
|------|-----------------|----------|
| `task_prompt` | Prompt Structure, Continuation | Task prompts, evaluation items |
| `generic_aigt` | All four channels | Reports, essays, expository text |
| `auto` | Heuristic selection | Default — detects mode from text |

### Determination Levels

| Level | Meaning | Action |
|-------|---------|--------|
| RED | Strong evidence of LLM generation | Flag for review, likely reject |
| AMBER | Substantial evidence, high confidence | Flag for manual review |
| MIXED | Conflicting strong signals across channels | Flag for manual review |
| YELLOW | Minor signals or convergence pattern | Note for awareness, may be legitimate |
| REVIEW | Weak sub-threshold signals worth noting | Optional manual review |
| GREEN | No significant signals detected | Pass |

### Conformal Calibration

When baseline data is available, the pipeline applies conformal calibration to raw confidence scores. The `conformity_level` field indicates how typical a confidence score is among calibrated human-authored texts (1.0 = typical of human text, 0.01 = very unusual).

### Short-Text Handling

For texts under 100 words, many analyzers bail out (NSSI < 150w, continuation < 80w, perplexity < 50w, TOCSIN < 40w). The fusion layer detects when fewer than 2 channels can produce scores and relaxes multi-channel corroboration requirements with a 0.15 confidence penalty. This prevents short texts from always appearing GREEN simply because most channels couldn't run. The L0 CRITICAL path (preamble) is unaffected.

### Channel Ablation

For diagnostic and evaluation purposes, individual channels can be disabled:

```bash
# Disable stylometric and continuation channels
python -m llm_detector --text "..." --disable-channel stylometric,continuation

# Run prompt structure only
python -m llm_detector input.xlsx --disable-channel stylometric,continuation,windowed
```

Disabled channels appear as GREEN no-ops in the audit trail. This is useful for measuring per-channel contribution to detection accuracy.

### Attack-Type Tagging

When collecting baseline data with `--collect`, the pipeline automatically derives an `attack_type` field from normalization signals: `homoglyph`, `zero_width`, `encoding`, `combined`, or `none`. This enables per-attack-type degradation analysis without changes to the detector itself.

### Memory System (BEET)

The pipeline includes a persistent memory store (`.beet/` directory) that tracks submissions, attempter profiles, and confirmed labels across batches. This enables cross-batch similarity detection, attempter risk profiling, and empirical calibration from reviewer feedback.

```bash
# First batch — initializes memory
python -m llm_detector input.xlsx --memory .beet/

# Second batch — automatically compares against first
python -m llm_detector input2.xlsx --memory .beet/

# Confirm labels from human review
python -m llm_detector --memory .beet/ --confirm task_001 ai reviewer_A
python -m llm_detector --memory .beet/ --confirm task_002 human reviewer_B

# Check an attempter's history
python -m llm_detector --memory .beet/ --attempter-history worker_42

# View memory summary
python -m llm_detector --memory .beet/ --memory-summary

# Rebuild calibration from confirmed labels
python -m llm_detector --memory .beet/ --rebuild-calibration

# Use rebuilt calibration for next batch
python -m llm_detector input3.xlsx --memory .beet/ --cal-table .beet/calibration.json
```

#### Attempter Risk Tiers

The memory store assigns risk tiers to contributors based on flag rate and confirmation history:

| Tier | Criteria |
|------|----------|
| **CRITICAL** | `confirmed_ai > 0` AND `flag_rate > 0.50` |
| **HIGH** | `flag_rate > 0.30` OR `confirmed_ai > 0` |
| **ELEVATED** | `flag_rate > 0.15` |
| **NORMAL** | `flag_rate <= 0.15` |

#### Memory Store Files

| File | Purpose |
|------|---------|
| `config.json` | Store metadata: version, submission counts, occupations |
| `submissions.jsonl` | All scored submissions with full feature vectors |
| `fingerprints.jsonl` | MinHash + optional embedding vectors for similarity |
| `attempters.jsonl` | Per-contributor aggregated history and risk profiles |
| `confirmed.jsonl` | Human-verified ground truth labels (feedback loop) |
| `calibration.json` | Current conformal calibration table |
| `shadow_model.pkl` | Trained shadow classifier (from `--rebuild-shadow`) |
| `shadow_disagreements.jsonl` | Logged rule/model disagreements for review |
| `centroids/` | Versioned semantic centroids (from `--rebuild-centroids`) |
| `lexicon_discovery/` | Lexicon candidate CSVs (from `--discover-lexicon`) |
| `calibration_history/` | Historical calibration snapshots |

### ML Tools

Three learning tools operate on confirmed labels to improve detection over time. All produce artifacts for human review — none modify the primary rule engine automatically.

#### Shadow Model (Parallel Corroboration)

An L1-penalized logistic regression trained on confirmed labels that runs alongside the rule engine. Flags *disagreements* — either rule-engine blind spots (model sees AI, rules say GREEN) or possible false positives (rules say RED, model sees human).

```bash
python -m llm_detector --memory .beet/ --rebuild-shadow
```

When active, shadow disagreements appear in pipeline output:

```
     ⚠️ SHADOW: Possible blind spot — learned model detects AI patterns that rule engine misses
         Rule=GREEN, Model=92.3% AI
```

Requires >= 200 confirmed labels with >= 30 per class.

#### Smoothed Log-Odds Lexicon Discovery

Discovers vocabulary disproportionately used in confirmed AI text vs human text using Monroe et al. (2008) informative Dirichlet prior. Outputs candidates to a CSV for human review.

```bash
python -m llm_detector --memory .beet/ --discover-lexicon \
    --labeled-corpus labeled_prompts.jsonl
```

Where `labeled_prompts.jsonl` contains `{"task_id": "...", "text": "..."}` lines.

#### Versioned Semantic Centroid Rebuilder

Recomputes AI and human semantic centroids from confirmed labeled text, replacing the hardcoded 5-sentence archetypes with data-driven centroids using k-means clustering.

```bash
python -m llm_detector --memory .beet/ --rebuild-centroids \
    --labeled-corpus labeled_prompts.jsonl
```

#### Unified Rebuild

Rebuild all learned artifacts in one command:

```bash
python -m llm_detector --memory .beet/ --rebuild-all \
    --labeled-corpus labeled_prompts.jsonl
```

This runs calibration, shadow model, centroid rebuild, and lexicon discovery in sequence.

#### Feedback Loop

The memory system creates a complete feedback cycle: batch scoring produces submissions and disagreement logs, human reviewers confirm labels, and rebuild commands update all learned artifacts for subsequent batches.

```
  Batch Run               Human Review            Rebuild
  ─────────               ────────────            ───────
  python -m llm_detector  Reviewer confirms       python -m llm_detector
    input.xlsx            ground truth labels       --memory .beet/
    --memory .beet/              │                  --rebuild-all
           │                     ▼                  --labeled-corpus texts.jsonl
           ▼              .beet/confirmed.jsonl            │
  .beet/submissions.jsonl                                  ▼
  .beet/fingerprints.jsonl ─────────────────> .beet/calibration.json
  .beet/attempters.jsonl                      .beet/shadow_model.pkl
  .beet/shadow_disagreements.jsonl            .beet/centroids/
           │                                  .beet/lexicon_discovery/
           └───── next batch uses updated artifacts ◄──────┘
```

The shadow model and calibration rebuild use scalar features from `submissions.jsonl` — they don't need raw text. Centroids and lexicon discovery operate on embeddings and word frequencies, so they require `--labeled-corpus`.

| Operation | Needs `--labeled-corpus`? | Needs `--memory`? |
|---|:---:|:---:|
| Batch scoring | No | Optional |
| Record batch to memory | No | Yes |
| Cross-batch similarity | No | Yes |
| Attempter profiling | No | Yes |
| Confirm a label | No | Yes |
| Rebuild calibration | No | Yes |
| Rebuild shadow model | No | Yes |
| Rebuild centroids | **Yes** | Yes |
| Discover lexicon | **Yes** | Yes |

## Package Structure

```
llm_detector/                  # Main package
    __init__.py                # Version, public API re-exports
    __main__.py                # python -m llm_detector entry point
    compat.py                  # Feature flags (HAS_SPACY, HAS_FTFY, etc.)
    text_utils.py              # Shared utilities
    normalize.py               # Text normalization
    language_gate.py           # Language/fairness support check
    pipeline.py                # Full pipeline orchestration
    fusion.py                  # Evidence fusion across channels
    calibration.py             # Conformal calibration
    baselines.py               # Baseline collection and analysis
    similarity.py              # Cross-submission similarity
    memory.py                  # BEET memory store + ML tools
    io.py                      # File I/O (XLSX, CSV, PDF)
    cli.py                     # Command-line interface
    gui.py                     # Desktop GUI
    dashboard.py               # Streamlit web dashboard
    html_report.py             # HTML report generation
    reporting.py               # Report formatting

    analyzers/                 # One module per detection layer
        preamble.py
        fingerprint.py
        prompt_signature.py
        voice_dissonance.py
        instruction_density.py
        semantic_resonance.py
        self_similarity.py
        continuation_api.py
        continuation_local.py
        perplexity.py
        stylometry.py
        semantic_flow.py
        windowing.py
        token_cohesiveness.py

    channels/                  # Channel scoring
        prompt_structure.py
        stylometric.py
        continuation.py
        windowed.py

    lexicon/                   # Externalized detection vocabulary
        packs.py               # LexiconPack definitions & scoring engine
        integration.py         # Enhanced layer wrappers

.beet/                         # Memory store (created by --memory)
    config.json                # Store metadata and stats
    submissions.jsonl          # All scored submissions
    fingerprints.jsonl         # MinHash/structural fingerprints
    attempters.jsonl           # Attempter risk profiles
    confirmed.jsonl            # Human-confirmed ground truth labels
    calibration.json           # Empirical calibration table
    shadow_model.pkl           # Trained shadow classifier
    shadow_disagreements.jsonl # Logged shadow model disagreements
    centroids/                 # Versioned semantic centroids
    lexicon_discovery/         # Lexicon candidate CSVs
    calibration_history/       # Historical calibration snapshots

tests/                         # Test suite
run_detector                   # Thin CLI launcher
```

## Installation

```bash
pip install .                    # Core only (pandas, openpyxl)
pip install ".[api]"             # + Anthropic/OpenAI for DNA-GPT continuation
pip install ".[nlp]"             # + spaCy, sentence-transformers, scikit-learn
pip install ".[perplexity]"      # + transformers/torch for perplexity scoring
pip install ".[pdf]"             # + pypdf for PDF input
pip install ".[all]"             # Everything
```

Or install dependencies individually:

```bash
pip install openpyxl pandas
# Optional (improves sentence segmentation):
pip install spacy
# Optional (semantic resonance layer):
pip install sentence-transformers scikit-learn
# Optional (perplexity scoring):
pip install transformers torch
# Optional (robust Unicode normalization):
pip install ftfy
# Optional (PDF input):
pip install pypdf
# Optional (DNA-GPT API continuation):
pip install anthropic  # or: pip install openai
```

## Usage

### Single Text Analysis

```bash
python -m llm_detector --text "Your prompt text here"
# or
./run_detector --text "Your prompt text here"
```

### Desktop GUI

```bash
python -m llm_detector --gui
```

The desktop interface uses a dashboard-style layout (tabbed workspace, card sections, top-line KPI metrics in the Analysis tab, and improved visual hierarchy) so it feels closer to a modern analytics UI while keeping the existing Tkinter workflow and features.

### Web GUI (Streamlit)

Install Streamlit first if you haven't already:

```bash
pip install streamlit
```

Then launch the web dashboard with any of the following:

```bash
# Via the dedicated entry point (recommended after pip install)
llm-detector-dashboard

# Via the --web flag on the main CLI
python -m llm_detector --web

# Direct Streamlit invocation
streamlit run llm_detector/dashboard.py
```

The web dashboard opens in your browser and provides the same seven pages as the desktop GUI (Analysis, Configuration, Memory & Learning, Calibration, Reports, Quick Reference, Precheck), without requiring a local display.

### File Mode (XLSX/CSV/PDF)

```bash
python -m llm_detector input.xlsx --sheet "Sheet1" --prompt-col "prompt"
python -m llm_detector input.csv --prompt-col "content"
python -m llm_detector document.pdf
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--text` | Analyze a single text string |
| `--gui` | Launch desktop GUI mode |
| `--web` | Launch Streamlit web dashboard |
| `--sheet` | Sheet name for XLSX files |
| `--prompt-col` | Column name containing prompts (default: "prompt") |
| `--verbose`, `-v` | Show all layer details for every result |
| `--output`, `-o` | Output CSV path |
| `--attempter` | Filter by attempter name (substring match) |
| `--no-similarity` | Skip cross-submission similarity analysis |
| `--similarity-threshold` | Jaccard threshold for similarity flagging (default: 0.40) |
| `--similarity-store JSONL` | Path to persistent similarity store JSONL |
| `--no-layer3` | Skip continuation analysis entirely |
| `--disable-channel CHANNELS` | Comma-separated channel names to disable for ablation (`prompt_structure`, `stylometric`, `continuation`, `windowed`) |
| `--api-key` | API key for DNA-GPT continuation analysis |
| `--provider` | LLM provider: `anthropic` or `openai` (default: anthropic) |
| `--dna-model MODEL` | Override the DNA-GPT model name |
| `--ppl-model MODEL` | Override the perplexity model name |
| `--dna-samples N` | Number of continuation samples per truncation (default: 3) |
| `--workers N` | Number of parallel worker threads for batch processing (default: 1) |
| `--batch` | Enable batch processing mode (parallelizes file-level runs) |
| `--mode` | Detection mode: `task_prompt`, `generic_aigt`, or `auto` (default: auto) |
| `--collect PATH` | Append scored results to JSONL for baseline accumulation |
| `--analyze-baselines JSONL` | Compute percentile distributions from accumulated data |
| `--baselines-csv PATH` | Export baseline analysis to CSV |
| `--calibrate JSONL` | Build calibration table from labeled baselines |
| `--cal-table JSON` | Path to calibration table JSON |
| `--cost-per-prompt N` | Estimated cost per prompt in base units for reporting (default: 400.0) |
| `--html-report FILE` | Generate HTML report to file path |
| `--instructions FILE` | Path to custom grading instructions file |
| `--run-dir DIR` | Root directory for this run; creates a timestamped subfolder and sets default paths for output, report, memory, and similarity store |
| `--memory DIR` | Enable BEET memory store at given directory |
| `--memory-summary` | Print memory store summary |
| `--confirm TASK_ID LABEL REVIEWER` | Record ground truth confirmation |
| `--attempter-history NAME` | Show historical profile for an attempter |
| `--rebuild-calibration` | Rebuild calibration table from confirmed labels |
| `--rebuild-shadow` | Rebuild shadow model from confirmed labels |
| `--discover-lexicon` | Run log-odds lexicon discovery on confirmed labels |
| `--rebuild-centroids` | Rebuild semantic centroids from confirmed labels |
| `--rebuild-all` | Rebuild all learned artifacts at once |
| `--labeled-corpus JSONL` | Path to JSONL with raw text (for lexicon/centroid tools) |
| `--label` | Interactive labeling mode: review results and assign ground truth labels |
| `--label-output JSONL` | JSONL path for labeled records |
| `--label-reviewer NAME` | Reviewer name/ID for labeling session |
| `--label-skip-green` | Skip GREEN determinations during labeling (assume correct) |
| `--label-skip-red` | Skip RED determinations during labeling (assume correct) |
| `--label-max N` | Maximum number of items to label per session |
| `--calibration-report JSONL` | Generate calibration diagnostics report from labeled JSONL |
| `--calibration-report-csv PATH` | Export labeled data to CSV (use with `--calibration-report`) |
| `--id-col COL` | Column name or letter (A–Z) for task ID (default: task_id) |
| `--occ-col COL` | Column name or letter (A–Z) for occupation/area (default: occupation) |
| `--attempter-col COL` | Column name or letter (A–Z) for attempter/author (default: attempter_name) |
| `--stage-col COL` | Column name or letter (A–Z) for pipeline stage (default: pipeline_stage_name) |
| `--attempter-email-col COL` | Column name or letter (A–Z) for attempter email (optional) |
| `--reviewer-col COL` | Column name or letter (A–Z) for reviewer name (optional) |
| `--reviewer-email-col COL` | Column name or letter (A–Z) for reviewer email (optional) |

### Memory System (BEET)

See the [Memory System (BEET)](#memory-system-beet) section above for full documentation on the BEET store, ML tools, attempter risk tiers, and store file reference.

```bash
# Run with memory enabled
python -m llm_detector input.xlsx --memory .beet/

# Confirm ground-truth labels after human review
python -m llm_detector --memory .beet/ --confirm task_001 ai reviewer_A
python -m llm_detector --memory .beet/ --confirm task_002 human reviewer_B

# Rebuild all learned artifacts from confirmed labels
python -m llm_detector --memory .beet/ --rebuild-all --labeled-corpus texts.jsonl
```

### Python API

```python
from llm_detector import analyze_prompt

result = analyze_prompt(
    text="You are a board-certified pharmacist. Analyze the following...",
    task_id="task_001",
    occupation="pharmacist",
    attempter="worker_42",
)

print(result['determination'])       # RED / AMBER / YELLOW / GREEN
print(result['reason'])              # Primary signal description
print(result['confidence'])          # 0.0 - 1.0
print(result['supporting_signals'])  # List of other signals that fired

# Layer-level diagnostics
print(result['voice_dissonance_vsd'])            # Voice-Specification Dissonance
print(result['prompt_signature_composite'])      # Prompt signature composite
print(result['instruction_density_idi'])         # Instruction Density Index

# Shadow model disagreement (when memory store is active)
if result.get('shadow_disagreement'):
    print(result['shadow_disagreement']['interpretation'])
    print(result['shadow_disagreement']['shadow_ai_prob'])

# Cross-submission similarity (batch mode)
from llm_detector import analyze_similarity
results = [analyze_prompt(t['prompt'], ...) for t in tasks]
text_map = {r['task_id']: t['prompt'] for r, t in zip(results, tasks)}
flags = analyze_similarity(results, text_map)
```

#### Memory Store API

```python
from llm_detector.memory import MemoryStore

store = MemoryStore('.beet/')

# Record a batch
store.record_batch(results, text_map)

# Query attempter history
history = store.get_attempter_history('worker_42')
print(history['profile']['risk_tier'])    # CRITICAL / HIGH / ELEVATED / NORMAL
print(history['profile']['flag_rate'])    # 0.0 - 1.0

# Cross-batch similarity
cross_flags = store.cross_batch_similarity(results, text_map)

# Record confirmed label
store.record_confirmation('task_001', 'ai', verified_by='reviewer_A')

# Rebuild learned artifacts
store.rebuild_shadow_model()
store.discover_lexicon_candidates('labeled_prompts.jsonl')
store.rebuild_semantic_centroids('labeled_prompts.jsonl')

# Run shadow model on a result
disagreement = store.check_shadow_disagreement(result)
```

## Testing

```bash
# Run all tests:
pytest -q

# Individual test files:
python tests/test_pipeline.py
python tests/test_analyzers.py
python tests/test_memory.py
python tests/test_continuation_local.py
python tests/test_windowed.py
python tests/test_token_cohesiveness.py
python tests/test_fusion.py
python tests/test_fusion_edge_cases.py
python tests/test_normalize.py
python tests/test_similarity.py
python tests/test_reporting.py
python tests/test_html_report.py
python tests/test_calibration.py
python tests/test_lexicon.py
python tests/test_preamble_cot.py
python tests/test_xray_spans.py
python tests/test_cli.py
python tests/test_baselines_collection.py
python tests/test_channels_prompt_structure.py
python tests/test_channels_stylometric.py
python tests/test_dna_truncation.py
python tests/test_gui_dashboard_features.py
python tests/test_io_loaders.py
python tests/test_mattr_burstiness.py
python tests/test_ml_fusion.py
python tests/test_ml_mocked.py
python tests/test_prompt_structure_channel.py
python tests/test_semantic_flow.py
python tests/test_stylometric_channel.py
```

## Design Principles

**Density over presence.** Individual prompt-engineering patterns (role-setting, format directives) are expected in human-authored prompts. The signal is the *density* and formulaic stacking of multiple categories — not any single pattern.

**No single-layer vetoes.** Every layer can be defeated individually. The convergence floor ensures that when multiple layers whisper, the system still listens.

**Voice gate preserves specificity.** Voice Dissonance requires actual casual voice absence, not just specification presence. A human writing formal text naturally varies more than an LLM generating sterile specifications.

**Diagnostic layers inform but don't trigger.** Fingerprint analysis participates in convergence and similarity analysis but doesn't fire standalone signals — the false positive rate on individual vocabulary items is too high.

**Audit trail by default.** Every determination includes the primary signal, all supporting signals, and full layer-level diagnostics. Nothing is hidden from the reviewer.

**ML tools advise, never decide.** The shadow model, lexicon discovery, and centroid rebuilder produce artifacts for human review. They never modify the primary rule engine automatically — the human reviewer remains the decision-maker in the feedback loop.

## Documentation

Detailed step-by-step documentation for every component is available in the [`docs/`](docs/) directory:

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture.md) | System design, data flow diagrams, module dependency graph |
| [Pipeline Orchestration](docs/pipeline.md) | Step-by-step walkthrough of `analyze_prompt()` |
| [Detection Analyzers](docs/analyzers.md) | All 13+ analyzers with signals, thresholds, and academic references |
| [Scoring Channels](docs/channels.md) | How analyzers are aggregated into 4 independent channels |
| [Evidence Fusion](docs/fusion.md) | Priority logic, convergence rules, and ML fusion |
| [Lexicon Pack System](docs/lexicon.md) | 16 vocabulary families and integration layer |
| [Infrastructure Components](docs/infrastructure.md) | I/O, calibration, baselines, similarity, memory, normalization |
| [User Interfaces](docs/interfaces.md) | CLI, GUI, and Dashboard usage guide |
| [Configuration & Dependencies](docs/configuration.md) | Feature flags, optional dependencies, settings |
| [Testing Guide](docs/testing.md) | Test conventions, running tests, and coverage |

## Release Roadmap

| Version | Focus |
|---------|-------|
| v0.61 | Lexicon wiring, calibration fixes, test coverage |
| v0.62 | CI, lazy loading, MIXED/REVIEW, naming, docs |
| v0.63 | CoT leakage, DivEye variance, fingerprints |
| v0.64 | Empirical calibration with labeled data |
| v0.65a | Continuation/windowed/compressibility features |
| v0.65b | Enhanced similarity (adaptive, semantic, cross-batch) |
| v0.66 | Span annotation, attempter profiling, reporting, HTML |
| v0.67 ✓ | Memory system + ML tools (shadow model, lexicon discovery, centroids) |
| **v0.68** ✓ | **Semantic flow analysis, topic-scrubbed stylometry, interactive labeling, run-directory management, expanded column mapping** |

## License

MIT
