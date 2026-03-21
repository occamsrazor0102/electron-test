# Evidence Fusion

> How channel results are combined into a final determination using priority-based aggregation, convergence logic, and optional ML fusion.

**Module:** `llm_detector/fusion.py`
**ML extension:** `llm_detector/ml_fusion.py`

---

## Overview

The fusion layer is the decision-making core of the pipeline. It takes the four [channel results](channels.md) and combines them into a single determination (RED/AMBER/MIXED/YELLOW/REVIEW/GREEN) with a confidence score and human-readable explanation.

**Key principles:**
1. **Priority-based** — higher-severity signals take precedence
2. **Convergence-rewarding** — multiple channels agreeing → confidence boost
3. **Mode-aware** — different channels are primary vs. supporting depending on detection mode
4. **Transparent** — every determination includes the specific rule that fired and full channel breakdowns

---

## Function Signature

```python
determine(
    preamble_score, preamble_severity,     # From preamble analyzer
    prompt_sig,                             # From prompt signature analyzer
    voice_dis,                              # From voice dissonance analyzer
    instr_density=None,                     # From instruction density analyzer
    word_count=0,                           # Text word count
    self_sim=None,                          # From self-similarity analyzer
    cont_result=None,                       # From continuation analyzer
    lang_gate=None,                         # Language support gate result
    norm_report=None,                       # Normalization report
    mode='auto',                            # Detection mode
    fingerprint_score=0.0,                  # From fingerprint analyzer
    semantic=None,                          # From semantic resonance
    ppl=None,                               # From perplexity
    tocsin=None,                            # From token cohesiveness
    disabled_channels=None,                 # Channels to ablate
    semantic_flow=None,                     # From semantic flow
    ml_fusion_enabled=False,                # Enable ML fusion
    ml_model_path=None,                     # Path to trained ML model
    **kwargs
)
```

**Returns:**
```python
(determination, reason, confidence, channel_details)
```

---

## Fusion Process — Step by Step

### Step 1: Mode Detection

The fusion layer first determines whether the text is a task prompt or general AI-generated text:

```
Auto-detect as task_prompt if:
    prompt_sig.composite ≥ 0.15
    OR instr_density.idi ≥ 5
    OR framing_completeness ≥ 2

Auto-detect as generic_aigt if:
    self_sim.nssi_signals ≥ 3
    OR word_count ≥ 400
```

The mode determines which channels are primary vs. supporting.

### Step 2: Channel Scoring

All four channels are scored independently:

| Channel | Function | Input Analyzers |
|---------|----------|----------------|
| Prompt Structure | `score_prompt_structure()` | Preamble, prompt sig, voice dis, IDI, SSI |
| Stylometric | `score_stylometric()` | NSSI, semantic, PPL, fingerprint, TOCSIN, semantic flow |
| Continuation | `score_continuation()` | DNA-GPT API or local |
| Windowed | `score_windowed()` | Window scores, hot spans |

Each returns a `ChannelResult` with score, severity, explanation, and mode eligibility.

### Step 3: Role Assignment

Based on the detected mode, channels are assigned roles:

| Channel | task_prompt | generic_aigt |
|---------|------------|--------------|
| Prompt Structure | **Primary** | Supporting |
| Stylometric | Supporting | **Primary** |
| Continuation | **Primary** | **Primary** |
| Windowed | — | Supporting |

### Step 4: Severity Counting

The fusion layer counts severities across all active channels:

```python
n_red = count of channels at RED
n_amber_plus = count of channels at AMBER or higher
n_yellow_plus = count of channels at YELLOW or higher
n_primary_red = count of PRIMARY channels at RED
n_primary_amber = count of PRIMARY channels at AMBER or higher
n_primary_yellow_plus = count of PRIMARY channels at YELLOW or higher
```

### Step 5: Priority-Based Aggregation

Rules are evaluated in strict priority order — the first rule that matches determines the outcome:

#### Rule L0: CRITICAL (Preamble Early Exit)
```
If preamble_severity == "CRITICAL" AND preamble_score ≥ 0.99:
    → RED at 0.99 confidence
    Triggering rule: "l0_critical_preamble"
```
This is an unconditional override — no other signals can contradict a CRITICAL preamble hit.

#### Rule 1: Primary RED with Corroboration
```
If primary_red ≥ 1 AND yellow_plus ≥ 2:
    → RED at max(primary_red_scores)
    Triggering rule: "primary_red_with_corroboration"
```

#### Rule 2: Multi-Primary AMBER
```
If primary_amber ≥ 2:
    → RED at mean(primary_amber_scores) + 0.10 convergence boost
    Triggering rule: "multi_primary_amber_convergence"
```

#### Rule 3: Short-Text RED
```
If word_count < 100 AND primary_red ≥ 1 AND yellow_plus ≥ 1:
    → RED at max(primary_red_scores) - 0.15 penalty
    Triggering rule: "short_text_primary_red"
```

#### Rule 4: Single Primary AMBER
```
If primary_amber ≥ 1:
    → AMBER at max(primary_amber_scores)
    Triggering rule: "single_primary_amber"
```

#### Rule 5: Multi-Channel YELLOW Convergence
```
If yellow_plus ≥ 2:
    → AMBER at mean(yellow_plus_scores) + 0.10 convergence boost
    Triggering rule: "multi_channel_yellow_convergence"
```

#### Rule 6: Supporting AMBER in task_prompt
```
If mode == task_prompt AND any supporting channel at AMBER:
    → AMBER at supporting_score
    Triggering rule: "supporting_amber_task_prompt"
```

#### Rule 7: Single YELLOW+
```
If yellow_plus ≥ 1:
    → YELLOW at max(yellow_plus_scores)
    Triggering rule: "single_yellow_plus"
```

#### Rule 8: Weak Signals
```
If any channel score > 0.05:
    → REVIEW at max(scores)
    Triggering rule: "weak_sub_threshold"
```

#### Rule 9: No Signals
```
Otherwise:
    → GREEN at 0.0
    Triggering rule: "no_signals"
```

### Step 6: Fairness Severity Cap

If the language gate reports:
- `UNSUPPORTED` → cap determination to YELLOW
- `REVIEW` → cap determination to AMBER (if currently higher)

This prevents false positives on non-English text.

### Step 7: Multi-Layer Convergence Boost

When multiple channels agree at the same severity:
- 2+ channels at RED/AMBER → +0.10 confidence boost
- This rewards agreement across independent detection methods

---

## Channel Details Output

The `channel_details` dict returned by fusion contains:

```python
{
    "mode": "task_prompt",
    "channels": {
        "prompt_structure": {
            "score": 0.85,
            "severity": "RED",
            "explanation": "Prompt signature composite 0.65 (RED)",
            "role": "primary",
            "data_sufficient": True,
            "disabled": False
        },
        "stylometric": { ... },
        "continuation": { ... },
        "windowed": { ... }
    },
    "triggering_rule": "primary_red_with_corroboration",
    "fusion_counts": {
        "n_red": 1,
        "n_amber_plus": 2,
        "n_yellow_plus": 3,
        "n_primary_red": 1,
        "n_primary_amber": 1,
        "n_primary_yellow_plus": 2
    },
    "active_channels": 4,
    "short_text_adjustment": False
}
```

---

## ML Fusion (Optional)

**Module:** `llm_detector/ml_fusion.py`

The ML fusion layer is an optional alternative to rule-based fusion. It trains a classifier on labeled baseline data and uses it to produce determinations.

### When It Activates
ML fusion is dormant by default. It activates when:
1. `ml_fusion_enabled=True` is passed to `determine()`
2. A trained model exists at `ml_model_path`
3. The model was trained on ≥ 200 confirmed labels with ≥ 30 per class

### Feature Extraction
`extract_fusion_features(result_dict)` extracts 67 numeric fields from the pipeline result:
- All per-analyzer scores (preamble, prompt_sig, voice_dis, IDI, NSSI, etc.)
- Sub-signal features (CFD, MFSR, VSD, BScore, PPL, etc.)
- Word count, compression ratio, function word ratio

### Training
```python
train_fusion_model(
    memory_store,              # MemoryStore with confirmed labels
    min_samples=200,           # Minimum total confirmed samples
    min_per_class=30,          # Minimum per class (ai/human)
    algorithm='gradient_boosting'  # or 'random_forest'
)
```

**Supported algorithms:**
| Algorithm | Config |
|-----------|--------|
| GradientBoosting | 200 estimators, max_depth=4, learning_rate=0.1 |
| RandomForest | 200 estimators, max_depth=8 |

**Training output:**
- Model saved to `store_dir/fusion_model.pkl`
- Returns: n_samples, n_ai, n_human, cv_auc, top_features

### ML Determination Thresholds
```
P(AI) ≥ 0.85 → RED
P(AI) ≥ 0.65 → AMBER
P(AI) ≥ 0.40 → YELLOW
P(AI) ≥ 0.15 → REVIEW
P(AI) < 0.15 → GREEN
```

### Shadow Disagreement Detection
When both rule-based and ML fusion run, the system detects disagreements:
- If rule-based says GREEN but ML says RED → flagged as shadow disagreement
- Useful for identifying cases where rule-based heuristics miss patterns that ML catches

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Scoring Channels](channels.md) — Channel scoring details
- [Detection Analyzers](analyzers.md) — Individual analyzer signals
- [Infrastructure: Calibration](infrastructure.md#calibration) — Post-fusion calibration
