# System Architecture

> How the LLM Authorship Signal Analyzer is structured, how data flows through it, and how each layer contributes to the final determination.

---

## High-Level Overview

The LLM Authorship Signal Analyzer is a multi-layer stylometric detection pipeline designed to identify LLM-generated or LLM-assisted text submitted through human data collection workflows. It is purpose-built for quality assurance in benchmark construction, clinical education assessments, and similar pipelines where human-authored content is expected.

**Core design principles:**

1. **No single signal is definitive** — the system combines evidence across 13+ independent analyzers.
2. **Graceful degradation** — optional heavy dependencies (PyTorch, spaCy, sentence-transformers) are guarded by feature flags; the pipeline works with any subset.
3. **Interpretability** — every determination includes character-level detection spans, per-signal breakdowns, and rule-based fusion audit trails.
4. **Mode awareness** — task prompts and general prose are analyzed differently, with mode-specific thresholds and channel eligibility.
5. **Fairness** — a language gate caps severity for non-English text to prevent false positives.

---

## Architecture Layers

The system is organized into five conceptual layers:

```
┌────────────────────────────────────────────────────────────────────┐
│                     User Interfaces (Layer 5)                      │
│            CLI  ·  Desktop GUI (Tkinter)  ·  Web Dashboard         │
├────────────────────────────────────────────────────────────────────┤
│                    Reporting & Persistence (Layer 4)                │
│    HTML Report · JSON/CSV Export · Memory Store · Baselines        │
├────────────────────────────────────────────────────────────────────┤
│                    Fusion & Calibration (Layer 3)                   │
│    Priority Fusion · ML Fusion · Conformal Calibration             │
├────────────────────────────────────────────────────────────────────┤
│                    Scoring Channels (Layer 2)                       │
│    Prompt Structure · Stylometric · Continuation · Windowed        │
├────────────────────────────────────────────────────────────────────┤
│                    Detection Analyzers (Layer 1)                    │
│    Preamble · Fingerprint · Prompt Signature · Voice Dissonance    │
│    Instruction Density · Self-Similarity · Continuation (API/Local)│
│    Semantic Resonance · Perplexity · Token Cohesiveness             │
│    Semantic Flow · Stylometry · Windowing                          │
├────────────────────────────────────────────────────────────────────┤
│                    Pre-Processing (Layer 0)                         │
│    Text Normalization · Language Gate · Lexicon Packs              │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Step-by-Step Processing Pipeline

```
Input Text
    │
    ▼
┌─────────────────────┐
│  1. Normalization    │  normalize.py
│     • ftfy repair    │  Detects homoglyphs, zero-width chars,
│     • Homoglyph fold │  encoding attacks. Returns clean text
│     • Invisible strip│  + detailed attack report.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Language Gate    │  language_gate.py
│     • FW coverage    │  Checks function-word coverage & non-Latin
│     • Script check   │  script ratio. Caps severity for unsupported
│     • Support level  │  languages (fairness protection).
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  3. Detection Analyzers (13 parallel/sequential)         │
│                                                          │
│  ┌──────────┐ ┌────────────┐ ┌─────────────────┐       │
│  │ Preamble │ │ Fingerprint│ │ Prompt Signature │       │
│  └──────────┘ └────────────┘ └─────────────────┘       │
│  ┌────────────────┐ ┌─────────────────────┐             │
│  │Voice Dissonance│ │Instruction Density  │             │
│  └────────────────┘ └─────────────────────┘             │
│  ┌─────────────────┐ ┌─────────────────────┐            │
│  │ Self-Similarity │ │ Semantic Resonance  │            │
│  └─────────────────┘ └─────────────────────┘            │
│  ┌────────────────────┐ ┌────────────────────┐          │
│  │ Continuation (API) │ │ Continuation (Local)│         │
│  └────────────────────┘ └────────────────────┘          │
│  ┌────────────┐ ┌───────────────────┐ ┌────────────┐   │
│  │ Perplexity │ │Token Cohesiveness │ │Semantic Flow│   │
│  └────────────┘ └───────────────────┘ └────────────┘   │
│  ┌────────────┐ ┌───────────┐                           │
│  │ Stylometry │ │ Windowing │                           │
│  └────────────┘ └───────────┘                           │
│                                                          │
│  + Lexicon Pack enhancement (16 vocabulary families)     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  4. Scoring Channels (4 independent channels)            │
│                                                          │
│  ┌────────────────┐  ┌──────────────┐                   │
│  │Prompt Structure│  │ Stylometric  │                   │
│  │  (task_prompt  │  │(generic_aigt │                   │
│  │   primary)     │  │  primary)    │                   │
│  └────────────────┘  └──────────────┘                   │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ Continuation │    │  Windowed    │                   │
│  │ (both modes) │    │(generic_aigt)│                   │
│  └──────────────┘    └──────────────┘                   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  5. Evidence Fusion                                      │
│     • Mode detection (task_prompt vs generic_aigt)       │
│     • Priority-based aggregation (RED > AMBER > YELLOW)  │
│     • Multi-layer convergence (agreement → boost)        │
│     • Optional ML fusion (LogisticRegression/RandomForest)│
│     • Shadow disagreement detection                       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  6. Conformal Calibration (optional)                     │
│     • Maps raw confidence → calibrated confidence        │
│     • Stratified by domain × length_bin                  │
│     • Returns conformity_level (1.0 → 0.01)             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  7. Result Assembly                                      │
│     • determination (RED/AMBER/MIXED/YELLOW/REVIEW/GREEN)│
│     • confidence (0.0–1.0)                               │
│     • reason (human-readable explanation)                 │
│     • detection_spans (character-level highlights)        │
│     • Per-analyzer scores + per-channel scores            │
│     • Calibration info + language gate status             │
└─────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
pipeline.py (orchestrator)
├── normalize.py
├── language_gate.py → text_utils.py
├── analyzers/*
│   ├── preamble.py
│   ├── fingerprint.py
│   ├── prompt_signature.py
│   ├── voice_dissonance.py
│   ├── instruction_density.py
│   ├── self_similarity.py
│   ├── continuation_api.py
│   ├── continuation_local.py
│   ├── semantic_resonance.py  → compat.py (HAS_SEMANTIC)
│   ├── perplexity.py          → compat.py (HAS_PERPLEXITY)
│   ├── token_cohesiveness.py  → compat.py (HAS_SEMANTIC)
│   ├── semantic_flow.py       → compat.py (HAS_SEMANTIC)
│   ├── stylometry.py
│   └── windowing.py
├── lexicon/*
│   ├── packs.py               (16 vocabulary family definitions)
│   └── integration.py         (enhanced analyzer wrappers)
├── fusion.py
│   ├── channels/*
│   │   ├── prompt_structure.py
│   │   ├── stylometric.py
│   │   ├── continuation.py
│   │   └── windowed.py
│   └── ml_fusion.py           (optional ML classifier)
├── calibration.py
└── text_utils.py

io.py          → (task dicts into pipeline)
reporting.py   → (reads pipeline result dict)
html_report.py → (reads pipeline result dict)
baselines.py   → (writes JSONL from results)
similarity.py  → (reads results + text_map)
memory.py      → (persistence across batches)
    ├── similarity.py (MinHash fingerprints)
    └── compat.py (HAS_SEMANTIC for embeddings)
```

---

## Detection Modes

The pipeline supports two detection modes, each optimizing for different text types:

| Mode | Primary Channels | Best For |
|------|-----------------|----------|
| `task_prompt` | Prompt Structure, Continuation | Task prompts, evaluation items, specification-heavy text |
| `generic_aigt` | All four channels | Reports, essays, expository prose |
| `auto` (default) | Heuristic detection | Automatically selects mode based on text characteristics |

**Auto-detection heuristics:**
- **task_prompt** if: prompt signature composite ≥ 0.15, OR instruction density IDI ≥ 5, OR framing completeness ≥ 2
- **generic_aigt** if: self-similarity signals ≥ 3, OR word count ≥ 400

---

## Determination Levels

The final output is one of six severity levels:

| Level | Confidence | Meaning | Recommended Action |
|-------|-----------|---------|-------------------|
| **RED** | 0.85–0.99 | Strong evidence of LLM generation | Flag for review, likely reject |
| **AMBER** | 0.65–0.85 | Substantial evidence, high confidence | Flag for manual review |
| **MIXED** | 0.60–0.80 | Conflicting strong signals across channels | Flag for manual review |
| **YELLOW** | 0.30–0.65 | Minor signals or convergence pattern | Note for awareness |
| **REVIEW** | 0.15–0.30 | Weak sub-threshold signals worth noting | Optional manual review |
| **GREEN** | 0.00–0.15 | No significant signals detected | Pass |

---

## Key Design Patterns

### 1. Graceful Degradation
Every analyzer checks feature availability (`HAS_SEMANTIC`, `HAS_PERPLEXITY`, etc.) and returns zero/empty results with an explanatory reason string if the dependency is unavailable. The pipeline never fails due to a missing optional dependency.

### 2. Priority-Based Fusion
- **L0 CRITICAL**: Preamble at 0.99 confidence + CRITICAL severity → immediate RED
- Channel severities ranked: RED > AMBER > YELLOW > GREEN
- Multi-layer convergence (multiple channels agreeing) → confidence boost

### 3. Externalized Vocabulary
Lexicon packs are frozen dataclass instances versioned independently of the analyzers they feed. Per-pack weight and cap settings prevent any single vocabulary family from dominating the overall score. New packs can be added without modifying analyzer code.

### 4. Conformal Calibration
Non-parametric quantile tables derived from labeled baseline data. No assumption of distribution shape. Returns interpretable conformity levels: 1.0 (typical of human text), 0.10 (unusual), 0.05 (very unusual), 0.01 (strong AI signal).

### 5. Short-Text Handling
For texts under 100 words, many analyzers bail out gracefully. The fusion layer detects when fewer than 2 channels can produce scores and relaxes multi-channel corroboration requirements with a 0.15 confidence penalty.

---

## Related Documentation

- [Pipeline Orchestration](pipeline.md) — Detailed walkthrough of `analyze_prompt()`
- [Detection Analyzers](analyzers.md) — All 13 analyzers with signals, thresholds, and academic references
- [Scoring Channels](channels.md) — How analyzers are aggregated into channels
- [Evidence Fusion](fusion.md) — Priority logic, convergence rules, and ML fusion
- [Lexicon Pack System](lexicon.md) — 16 vocabulary families and integration
- [Infrastructure Components](infrastructure.md) — I/O, calibration, baselines, similarity, memory, normalization
- [User Interfaces](interfaces.md) — CLI, GUI, and Dashboard usage
- [Configuration & Dependencies](configuration.md) — Feature flags, optional dependencies, settings
- [Testing Guide](testing.md) — Test conventions, running tests, and coverage
