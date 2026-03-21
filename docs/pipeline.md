# Pipeline Orchestration

> Detailed walkthrough of the core `analyze_prompt()` function — the single entry point for all text analysis.

**Module:** `llm_detector/pipeline.py`

---

## Overview

The pipeline orchestrates all detection layers in a fixed sequence: normalization → language gating → 13 analyzers → channel scoring → fusion → calibration → result assembly. It is the only module that a caller needs to interact with for detection — everything else is internal.

---

## Function Signature

```python
analyze_prompt(
    text,                          # The text to analyze (required)
    task_id='',                    # Submission identifier
    occupation='',                 # Author's occupation/domain
    attempter='',                  # Author name
    stage='',                      # Pipeline stage name
    run_l3=True,                   # Enable expensive L3 analysis
    api_key=None,                  # API key for DNA-GPT continuation
    dna_provider='anthropic',      # LLM provider: 'anthropic' or 'openai'
    dna_model=None,                # Specific model override for continuation
    dna_samples=3,                 # Number of continuation samples (K)
    ground_truth=None,             # Ground truth label if known ('ai'/'human'/'unsure')
    language=None,                 # Override language detection
    domain=None,                   # Domain for calibration stratification
    mode='auto',                   # Detection mode: 'auto', 'task_prompt', 'generic_aigt'
    cal_table=None,                # Conformal calibration table (loaded from JSON)
    memory_store=None,             # MemoryStore instance for cross-batch tracking
    disabled_channels=None,        # List of channel names to ablate
    precomputed_continuation=None, # Skip continuation if already computed
    ppl_model=None                 # Override perplexity model ID
)
```

**Returns:** A single dictionary with 150+ fields containing the full analysis result.

---

## Step-by-Step Processing

### Step 1: Text Normalization

```
Input text → normalize_text(text) → (normalized_text, delta_report)
```

**What happens:**
1. Applies `ftfy` encoding repair (if available)
2. Strips invisible characters (zero-width spaces, joiners)
3. Applies NFKC Unicode normalization
4. Folds homoglyphs (Cyrillic → ASCII, smart quotes → ASCII)
5. Collapses inter-character spacing ("l i k e" → "like")
6. Collapses excessive whitespace

**Output:** Cleaned text + a report documenting what was changed and what attack types were detected (homoglyph, zero_width, interspacing, encoding).

**Why it matters:** Cheap evasion attacks (inserting invisible characters, using Cyrillic lookalikes) would otherwise fool downstream analyzers. Normalization neutralizes these before any detection runs.

### Step 2: Language Gate

```
normalized_text → check_language_support(text, word_count) → gate_result
```

**What happens:**
1. Measures function-word coverage (ratio of top-50 English function words present)
2. Measures non-Latin script ratio
3. Assigns support level: SUPPORTED, REVIEW, or UNSUPPORTED

**Thresholds:**
- **UNSUPPORTED:** >30% non-Latin characters OR function-word coverage < 8%
- **REVIEW:** Function-word coverage 8–12% OR 10–30% non-Latin characters
- **SUPPORTED:** Function-word coverage ≥ 12% AND < 10% non-Latin characters

**Why it matters:** The detection layers are validated on English prose. Running them on non-English text produces unreliable results. The language gate caps severity (UNSUPPORTED → max YELLOW, REVIEW → max AMBER) to prevent false positives on non-English submissions.

### Step 3: Run All Analyzers

Each analyzer runs independently and produces its own score/result dict. Analyzers are run conditionally based on:
- Feature availability (e.g., semantic resonance requires `HAS_SEMANTIC`)
- The `run_l3` flag (self-similarity, continuation are "L3" — expensive)
- Text length minimums (many analyzers bail out on short text)

**Analyzer execution order:**

| # | Analyzer | Module | Min Length | Dependency |
|---|----------|--------|-----------|------------|
| 1 | Preamble | `preamble.py` | None | None |
| 2 | Fingerprint | `fingerprint.py` | None | None |
| 3 | Prompt Signature | `prompt_signature.py` | None | None (+ lexicon enhancement) |
| 4 | Voice Dissonance | `voice_dissonance.py` | None | None (+ lexicon enhancement) |
| 5 | Instruction Density | `instruction_density.py` | None | None (+ lexicon enhancement) |
| 6 | Self-Similarity (NSSI) | `self_similarity.py` | ~150 words | `run_l3=True` |
| 7 | Continuation (API) | `continuation_api.py` | ~80 words | `run_l3=True` + `api_key` |
| 8 | Continuation (Local) | `continuation_local.py` | ~80 words | `run_l3=True` |
| 9 | Semantic Resonance | `semantic_resonance.py` | ~50 words | `HAS_SEMANTIC` |
| 10 | Perplexity | `perplexity.py` | ~50 words | `HAS_PERPLEXITY` |
| 11 | Token Cohesiveness | `token_cohesiveness.py` | ~40 words | `HAS_SEMANTIC` |
| 12 | Semantic Flow | `semantic_flow.py` | ~5 sentences | `HAS_SEMANTIC` |
| 13 | Stylometry | `stylometry.py` | None | None |
| 14 | Windowing | `windowing.py` | ~5 sentences | None |

### Step 4: Detection Span Aggregation

After all analyzers run, the pipeline collects character-level detection spans from every source. Each span records:

```python
{
    "start": 42,       # Start character offset
    "end": 78,         # End character offset
    "text": "...",     # Matched text
    "source": "preamble",  # Which analyzer flagged it
    "label": "assistant_ack",  # Specific pattern name
    "type": "CRITICAL"  # Signal severity
}
```

These spans enable the HTML report to highlight exactly which parts of the text triggered which signals.

### Step 5: Evidence Fusion

```
All analyzer results → determine(...) → (determination, reason, confidence, channel_details)
```

The fusion layer:
1. Auto-detects mode (task_prompt vs. generic_aigt)
2. Scores all 4 channels
3. Applies priority-based aggregation with convergence rules
4. Optionally runs ML fusion for comparison
5. Detects shadow disagreement between rule-based and ML results

See [Evidence Fusion](fusion.md) for full details.

### Step 6: Conformal Calibration (Optional)

```
raw_confidence + cal_table → apply_calibration(...) → calibration_result
```

If a calibration table is provided (from labeled baseline data), the raw confidence is mapped to a calibrated confidence with a conformity level indicating how typical the score is among known human-authored texts.

See [Infrastructure: Calibration](infrastructure.md#calibration) for full details.

### Step 7: Result Assembly

The pipeline packages everything into a single result dictionary:

```python
{
    # Top-level determination
    "determination": "RED",          # RED/AMBER/MIXED/YELLOW/REVIEW/GREEN
    "confidence": 0.92,              # 0.0–1.0
    "reason": "Preamble CRITICAL + multi-channel convergence",
    "mode": "task_prompt",           # Detected or forced mode

    # Metadata
    "task_id": "task_001",
    "occupation": "data_annotator",
    "attempter": "worker_42",
    "stage": "prompt_creation",
    "word_count": 347,
    "version": "0.68.0",

    # Per-analyzer scores
    "preamble_score": 0.99,
    "preamble_severity": "CRITICAL",
    "fingerprint_score": 0.34,
    "prompt_sig": { ... },           # CFD, MFSR, composite, meta_design_hits
    "voice_dis": { ... },            # voice_score, spec_score, VSD
    "instr_density": { ... },        # IDI, imperatives, conditionals
    "self_similarity": { ... },      # nssi_score, nssi_signals, determination
    "continuation": { ... },         # bscore, multi_bscores, stability
    "semantic": { ... },             # ai_score, human_score, delta
    "perplexity": { ... },           # ppl, surprisal_variance, binoculars_score
    "tocsin": { ... },               # cohesiveness, cohesiveness_std
    "semantic_flow": { ... },        # flow_variance, flow_mean
    "windowing": { ... },            # max_window, hot_span, mixed_signal
    "stylometry": { ... },           # mattr, fw_ratio, sent_dispersion

    # Detection spans
    "detection_spans": [ ... ],      # Character-level highlight list

    # Channel details
    "channel_details": { ... },      # Per-channel scores, roles, fusion info

    # Calibration (if available)
    "calibrated_confidence": 0.94,
    "conformity_level": 0.01,
    "stratum_used": "data_annotator/medium",

    # Normalization report
    "norm_report": { ... },          # obfuscation_delta, attack_types

    # Language gate
    "lang_gate": { ... },            # support_level, fw_coverage
}
```

---

## Key Parameters Explained

### `run_l3` (Default: `True`)
Controls whether expensive "Layer 3" analyses run:
- **Self-Similarity (NSSI):** CPU-intensive formulaic pattern analysis
- **Continuation (DNA-GPT):** Requires LLM API call (API mode) or n-gram model building (local mode)

Set to `False` for fast screening passes where only structural signals are needed.

### `mode` (Default: `'auto'`)
- **`auto`:** Pipeline auto-detects based on text characteristics
- **`task_prompt`:** Forces task-prompt mode — Prompt Structure channel is primary
- **`generic_aigt`:** Forces generic mode — all four channels are active

### `disabled_channels`
A list of channel names to ablate for diagnostic purposes:
```python
analyze_prompt(text, disabled_channels=["stylometric", "continuation"])
```
Disabled channels appear as GREEN no-ops in the audit trail.

### `memory_store`
A `MemoryStore` instance for cross-batch tracking:
- Records submission to persistent store
- Computes MinHash fingerprints for similarity detection
- Checks shadow disagreement between primary and secondary models

### `cal_table`
A calibration table loaded from JSON (produced by `calibrate_from_baselines()`). When provided, the raw confidence is mapped to a calibrated confidence with conformity levels.

---

## Usage Examples

### Basic single-text analysis
```python
from llm_detector import analyze_prompt

result = analyze_prompt("Your text here")
print(result["determination"])  # RED, AMBER, YELLOW, GREEN, etc.
print(result["confidence"])     # 0.0–1.0
print(result["reason"])         # Human-readable explanation
```

### With calibration and memory
```python
from llm_detector import analyze_prompt, load_calibration
from llm_detector.memory import MemoryStore

cal_table = load_calibration("baselines/calibration.json")
memory = MemoryStore(".beet")

result = analyze_prompt(
    text="Your text here",
    task_id="task_001",
    occupation="annotator",
    cal_table=cal_table,
    memory_store=memory,
)
```

### Fast screening (no expensive analyses)
```python
result = analyze_prompt("Your text here", run_l3=False)
```

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Detection Analyzers](analyzers.md) — Individual analyzer details
- [Scoring Channels](channels.md) — Channel aggregation
- [Evidence Fusion](fusion.md) — Determination logic
