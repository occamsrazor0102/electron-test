# Scoring Channels

> How individual analyzer signals are aggregated into four independent scoring channels.

**Module directory:** `llm_detector/channels/`

---

## Overview

Scoring channels sit between the [detection analyzers](analyzers.md) and the [evidence fusion](fusion.md) layer. Each channel aggregates a specific subset of analyzer outputs into a single channel score with severity, explanation, and mode eligibility.

**Key principles:**
- Each channel operates independently — channels don't see each other's results
- Channels have mode eligibility — not all channels are active in every detection mode
- Each channel reports data sufficiency — if input text is too short for its analyzers, the channel reports `data_sufficient=False`
- Channel results are combined by the fusion layer using priority-based aggregation

---

## Channel Result Structure

Every channel produces a `ChannelResult` with:

```python
{
    "score": 0.0–1.0,           # Channel confidence score
    "severity": "RED",           # RED / AMBER / YELLOW / GREEN
    "explanation": "...",        # Human-readable reason
    "role": "primary",           # primary / supporting / disabled / no_data
    "mode_eligibility": [...],   # Which modes can use this channel
    "sub_signals": { ... },      # Breakdown of contributing analyzers
    "data_sufficient": True      # Whether the text was long enough
}
```

---

## Channel 1: Prompt Structure

**Module:** `channels/prompt_structure.py`
**Mode eligibility:** `task_prompt`, `generic_aigt`
**Role in task_prompt mode:** Primary
**Role in generic_aigt mode:** Supporting

### Purpose
Catches LLM-engineered task specifications — text that reads like something an LLM produced in response to "write me a task prompt." This is the primary detection channel for task_prompt mode.

### Aggregated Analyzers

| Analyzer | Signal | Weight |
|----------|--------|--------|
| **Preamble** | LLM output artifacts (critical early-exit) | Highest priority |
| **Prompt Signature** | CFD, MFSR, composite score | Primary structural signal |
| **Voice Dissonance** | VSD product (voice × spec) | Voice-gated threshold |
| **Instruction Density** | IDI score | Imperative/conditional density |
| **SSI (Sterile Spec Index)** | Spec without voice or hedges | Specification sterility |

### Scoring Logic

The channel evaluates signals in priority order:

**Step 1: Preamble Check (Critical Early Exit)**
```
If preamble severity == "CRITICAL":
    → Return RED, score=0.99, explanation="Preamble CRITICAL hit"
```

**Step 2: Prompt Signature Scoring**
```
composite ≥ 0.60 → RED (0.85)
composite ≥ 0.40 → AMBER (0.65)
composite ≥ 0.20 → YELLOW (0.40)
```

**Step 3: Voice Dissonance Scoring**
```
VSD ≥ 50 (voice-gated)  → RED (0.90)
VSD ≥ 21 (voice-gated)  → AMBER (0.70)
VSD ≥ 100 (ungated)     → AMBER (0.70)
VSD ≥ 21 (ungated)      → YELLOW (0.45)
```

**Step 4: Instruction Density Scoring**
```
IDI ≥ 12 → RED (0.85)
IDI ≥ 8  → AMBER (0.65)
```

**Step 5: Sterile Spec Index (SSI)**
```
spec ≥ 5–7, voice < 0.5, no hedges:
    SSI ≥ 8.0 → AMBER (0.70)
    else      → YELLOW (0.45)
```

**Step 6: Final Score**
The channel takes the maximum severity and score from all contributing signals.

---

## Channel 2: Stylometric

**Module:** `channels/stylometric.py`
**Mode eligibility:** `generic_aigt`
**Role in generic_aigt mode:** Primary
**Role in task_prompt mode:** Supporting

### Purpose
Detects LLM formulaic patterns in general prose through topic-invariant style analysis. This is the primary detection channel for generic_aigt mode.

### Aggregated Analyzers

| Analyzer | Signal | Role |
|----------|--------|------|
| **Self-Similarity (NSSI)** | Formulaic convergence (13 signals) | Primary |
| **Semantic Resonance** | Embedding proximity to AI centroids | Supporting boost |
| **Perplexity** | Low PPL, surprisal variance, binoculars | Supporting boost |
| **Token Cohesiveness (TOCSIN)** | Deletion stability | Supporting boost |
| **Semantic Flow** | Transition uniformity | Supporting boost |
| **Fingerprint** | Excess vocabulary hits | Conditional boost |
| **Surprisal variance/volatility** | DivEye + decay | Supplementary |
| **Perplexity burstiness** | Low sentence-PPL variance | Supplementary |

### Scoring Logic

**Step 1: NSSI Base Score (Primary)**
```
NSSI RED   → base_score = 0.85
NSSI AMBER → base_score = 0.65
NSSI YELLOW → base_score = 0.40
```

**Step 2: Supporting Boosts (Additive)**
```
Semantic resonance AMBER → +0.10
Semantic resonance YELLOW → +0.05
Perplexity AMBER → +0.10
Perplexity YELLOW → +0.05
TOCSIN AMBER → +0.10
TOCSIN YELLOW → +0.05
Semantic flow AMBER → +0.10
Semantic flow YELLOW → +0.05
Low burstiness → +0.06
```

**Step 3: Fingerprint Conditional Boost**
```
If severity ≠ GREEN AND fingerprint_score > 0:
    → +0.10
```

**Step 4: Data Sufficiency**
Requires at least 1 active analyzer (NSSI, semantic, PPL, TOCSIN, or fingerprints) to report `data_sufficient=True`.

---

## Channel 3: Continuation

**Module:** `channels/continuation.py`
**Mode eligibility:** `task_prompt`, `generic_aigt`
**Role:** Primary in both modes

### Purpose
Divergent continuation analysis — the most theoretically grounded detection method. Measures how much an LLM regeneration overlaps with the original text. Works in both task_prompt and generic_aigt modes.

### Aggregated Analyzers

| Analyzer | Signal | Condition |
|----------|--------|-----------|
| **Continuation API (DNA-GPT)** | BScore from LLM regeneration | Requires API key |
| **Continuation Local** | Composite from n-gram proxy | Always available |

### Scoring Logic

```
DNA-GPT determination:
    RED   → score = 0.95, severity = RED
    AMBER → score = 0.70, severity = AMBER
    YELLOW → score = 0.40, severity = YELLOW
    GREEN → score = 0.10, severity = GREEN
```

When both API and local continuation are available, the channel uses the API result (more accurate). Local serves as a zero-cost fallback.

### Data Sufficiency
Set to `False` if no continuation result is available (text too short, API not configured, or continuation disabled).

---

## Channel 4: Windowed

**Module:** `channels/windowed.py`
**Mode eligibility:** `generic_aigt`
**Role in generic_aigt mode:** Supporting

### Purpose
Detects mixed human+AI content through sentence-window analysis. Identifies concentrated "hot spans" of AI-like content within otherwise human-written text.

### Aggregated Analyzers

| Analyzer | Signal |
|----------|--------|
| **Windowing** | Per-window scores, hot spans, mixed signal, CUSUM changepoint |

### Scoring Logic

```
max_window ≥ 0.60 AND hot_span ≥ 3 → RED (0.75)
max_window ≥ 0.45 AND hot_span ≥ 2 → AMBER (0.55)
max_window ≥ 0.30                   → YELLOW (0.30)
```

When `mixed_signal` is detected AND severity ≠ GREEN, the explanation includes a note about potential mixed human+AI content.

### Data Sufficiency
Set to `False` if the text has fewer sentences than the window size (default: 5).

---

## Channel Interaction with Fusion

The four channel results flow into the [fusion layer](fusion.md) which:

1. **Determines mode** — auto-detects task_prompt vs. generic_aigt
2. **Assigns roles** — marks each channel as primary or supporting based on mode
3. **Counts severities** — tallies RED, AMBER, YELLOW across channels
4. **Applies priority rules** — L0 CRITICAL, primary RED + corroboration, multi-channel convergence
5. **Produces determination** — RED/AMBER/MIXED/YELLOW/REVIEW/GREEN

### Channel Roles by Mode

| Channel | task_prompt | generic_aigt |
|---------|------------|--------------|
| Prompt Structure | **Primary** | Supporting |
| Stylometric | Supporting | **Primary** |
| Continuation | **Primary** | **Primary** |
| Windowed | — | Supporting |

---

## Channel Ablation

Individual channels can be disabled for diagnostic purposes:

```bash
# Disable stylometric and continuation channels
llm-detector --text "..." --disable-channel stylometric,continuation

# Run prompt structure only
llm-detector input.xlsx --disable-channel stylometric,continuation,windowed
```

Disabled channels appear as GREEN no-ops (`role="disabled"`) in the channel details, with an explanation noting they were ablated.

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Detection Analyzers](analyzers.md) — Individual analyzer details
- [Evidence Fusion](fusion.md) — How channel results become determinations
- [Pipeline Orchestration](pipeline.md) — End-to-end processing flow
