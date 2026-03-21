# Infrastructure Components

> Supporting modules for I/O, calibration, baselines, similarity analysis, memory persistence, normalization, and language gating.

---

## Table of Contents

1. [File I/O Loaders](#file-io-loaders)
2. [Calibration](#calibration)
3. [Baseline Collection & Analysis](#baseline-collection--analysis)
4. [Cross-Submission Similarity](#cross-submission-similarity)
5. [Memory Store (BEET)](#memory-store-beet)
6. [Text Normalization](#text-normalization)
7. [Language Gate](#language-gate)
8. [Text Utilities](#text-utilities)
9. [Reporting](#reporting)
10. [HTML Report Generation](#html-report-generation)
11. [Compatibility & Feature Flags](#compatibility--feature-flags)
12. [Shared Constants](#shared-constants)

---

## File I/O Loaders

**Module:** `llm_detector/io.py`

Handles loading submissions from various file formats and extracting the text column with configurable column mapping.

### Functions

#### `load_xlsx(filepath, sheet=None, prompt_col='prompt', id_col='task_id', occ_col='occupation', attempter_col='attempter_name', stage_col='pipeline_stage_name', email_col=None, reviewer_col=None, reviewer_email_col=None)`

Loads submissions from an Excel file.

**Column specification options:**
- **Column header name** — matched case-insensitively with fuzzy substring fallback
- **Single letter (A–Z)** — positional reference, takes priority over name matching
- **1-based integer string** — e.g., '1' for the first column

**Features:**
- Auto-detects sheet from heuristics ('FullTaskX', 'Full Task Connected', etc.)
- Filters prompts to ≥50 characters (avoids blank/trivial rows)
- Returns list of task dicts: `{text, task_id, occupation, attempter, stage, ...}`

#### `load_csv(filepath, prompt_col='prompt', ...)`

Same interface as `load_xlsx` but reads CSV files via pandas with dialect auto-detection.

#### `load_pdf(filepath)`

Extracts text from PDF pages using pypdf:
- Each page becomes a separate task
- Pages with <50 characters of text are concatenated
- Returns list of task dicts

### How It Contributes
I/O loaders convert file inputs into the standardized task dict format consumed by `pipeline.analyze_prompt()`. The 7 configurable columns (task_id, occupation, attempter_name, pipeline_stage_name, attempter_email, reviewer, reviewer_email) are consistent across CLI, GUI, and dashboard.

---

## Calibration

**Module:** `llm_detector/calibration.py`

Implements conformal calibration from labeled baseline data, mapping raw confidence scores to calibrated confidence with interpretable conformity levels.

### How Conformal Calibration Works

1. **Build calibration table** from a collection of human-authored baseline texts with known labels
2. **Compute nonconformity scores:** `nc = 1.0 - confidence` for each human-labeled text
3. **Build quantile tables** at significance levels α = 0.01, 0.05, 0.10
4. **Stratify** by (domain, length_bin) if ≥10 samples per stratum
5. **Apply** to new texts: compare their nc score against the quantile table

### Functions

#### `calibrate_from_baselines(jsonl_path) → dict`

Builds calibration tables from a labeled baseline JSONL file.

**Input:** JSONL file where each line has at least `ground_truth`, `confidence`, `domain`, and `word_count`

**Output:**
```python
{
    "global": {0.01: threshold, 0.05: threshold, 0.10: threshold},
    "strata": {
        "annotator/medium": {0.01: ..., 0.05: ..., 0.10: ...},
        ...
    },
    "n_calibration": 250,
    "strata_counts": {"annotator/medium": 45, ...}
}
```

#### `apply_calibration(confidence, cal_table, domain=None, length_bin=None) → dict`

Maps a raw confidence score to calibrated confidence and conformity level.

**Conformity levels:**
| Level | Interpretation |
|-------|---------------|
| 1.0 | Typical of human-authored text |
| 0.10 | Unusual (10% significance) |
| 0.05 | Very unusual (5% significance) |
| 0.01 | Strong AI signal (1% significance) |

**Returns:**
```python
{
    "raw_confidence": 0.85,
    "calibrated_confidence": 0.91,
    "conformity_level": 0.01,
    "stratum_used": "annotator/medium"
}
```

#### `save_calibration(cal_table, output_path)` / `load_calibration(path)`

JSON serialization for calibration tables.

### Academic References
Vovk et al. (2005), Shafer & Vovk (2008) — Conformal prediction theory.

---

## Baseline Collection & Analysis

**Module:** `llm_detector/baselines.py`

Collects and analyzes labeled submission data for calibration and statistical analysis.

### Functions

#### `collect_baselines(results, output_path) → int`

Appends scored pipeline results to a JSONL file for baseline accumulation.

**Per-record fields (38 total):**
- Identification: task_id, occupation, attempter, stage
- Scores: preamble_score, fingerprint_score, prompt_sig_composite, voice_dis_vsd, idi, nssi_score, bscore, ppl, tocsin, semantic_flow_variance, ...
- Classification: determination, confidence, mode
- Metadata: word_count, version, timestamp, domain, length_bin
- Attack type: derived from normalization (homoglyph, zero_width, encoding, combined, none)

**Length bins:** short (<100 words), medium (<300), long (<800), very_long (≥800)

#### `analyze_baselines(jsonl_path, output_csv=None) → dict`

Reads a baseline JSONL file and computes:
- **Overall distribution:** RED/AMBER/YELLOW/GREEN counts and percentages
- **Per-occupation percentiles:** p25, p50, p75, p90, p95, p99 for 15 metrics
- **Labeled data analysis** (if available): TPR at FPR thresholds (1%, 5%, 10%)
- **Stratified analysis:** Flag rates by domain × length_bin
- **Fairness disparity detection:** Flags if max_rate - min_rate > 20 percentage points

### How It Contributes
Baselines provide the data for conformal calibration and enable statistical analysis of detection performance across occupations and domains.

---

## Cross-Submission Similarity

**Module:** `llm_detector/similarity.py`

Detects suspiciously similar submissions within the same occupation group — a signal of copy-paste, template reuse, or batch LLM generation.

### Features (FEAT 11–15)

#### FEAT 11: Adaptive Thresholds

```python
analyze_similarity(results, text_map, jaccard_threshold=0.40,
                   struct_threshold=0.90, semantic_threshold=0.92,
                   adaptive=True, instruction_text=None) → list
```

- Computes all pairwise similarities within occupation groups
- **Absolute thresholds** catch direct copy-paste
- **Adaptive thresholds** (occupation_median + 2σ) catch same-template generation
- Returns flagged pairs with z-scores

#### FEAT 12: Semantic Similarity

When `HAS_SEMANTIC` is available:
- Embeds full text using all-MiniLM-L6-v2
- Computes pairwise cosine similarity
- Adaptive threshold: `semantic_median + 2.0 × max(semantic_std, 0.01)`

#### FEAT 13: Similarity Feedback

```python
apply_similarity_adjustments(results, sim_pairs, text_map) → results
```

- Upgrades determination if submission is paired with semantic similarity + 2+ partners
- Never downgrades — only upgrades
- Tracks similarity context in audit trail

#### FEAT 14: MinHash Fingerprinting

```python
_shingle_fingerprint(shingle_set, n_hashes=128) → tuple
_minhash_similarity(fp_a, fp_b) → float  # Jaccard estimate
```

- 3-gram word shingles → 128-hash MinHash fingerprint
- MD5-based hash functions
- Compact storage: only 128 integers per submission
- `cross_batch_similarity()` compares current vs. historical fingerprints

#### FEAT 15: Instruction Template Factoring

Optionally removes shared instruction shingles before computing similarity, preventing false positives from identical task instructions that all submissions share.

---

## Memory Store (BEET)

**Module:** `llm_detector/memory.py`

Persistent memory system that tracks submissions, fingerprints, attempter profiles, and confirmed labels across batches. Stored in a `.beet/` directory.

### Class: `MemoryStore(store_dir='.beet')`

#### Key Methods

**`record_batch(results, text_map, batch_id=None)`**
Records a full batch to persistent storage:
- `submissions.jsonl` — all scored submissions
- `fingerprints.jsonl` — MinHash + optional semantic embeddings
- `attempters.jsonl` — per-contributor aggregated history

**`confirm_label(task_id, label, reviewer)`**
Records a human-verified ground truth label:
- Appends to `confirmed.jsonl`
- Updates attempter risk profile

**`get_attempter_history(attempter_name) → dict`**
Returns aggregated history for a contributor:
- Total submissions, flag rate, risk tier
- Per-determination counts
- Confirmed AI/human counts

**`check_shadow_disagreement(result) → dict`**
Compares primary determination against secondary model (if available) to detect conflicts.

### Persistent Files

| File | Format | Purpose |
|------|--------|---------|
| `config.json` | JSON | Store metadata: version, batch count, occupations |
| `submissions.jsonl` | JSONL | All scored submissions with full feature vectors |
| `fingerprints.jsonl` | JSONL | MinHash + embedding vectors for cross-batch similarity |
| `attempters.jsonl` | JSONL | Per-contributor aggregated history and risk profiles |
| `confirmed.jsonl` | JSONL | Human-verified ground truth labels |
| `calibration.json` | JSON | Conformal calibration table (rebuilt from confirmed labels) |

### Attempter Risk Tiers

| Tier | Criteria |
|------|----------|
| **CRITICAL** | confirmed_ai > 0 AND flag_rate > 0.50 |
| **HIGH** | flag_rate > 0.30 OR confirmed_ai > 0 |
| **ELEVATED** | flag_rate > 0.15 |
| **NORMAL** | flag_rate ≤ 0.15 |

---

## Text Normalization

**Module:** `llm_detector/normalize.py`

Pre-processing layer that neutralizes cheap evasion attacks before detection analyzers run.

### Function

```python
normalize_text(text) → (normalized_text, delta_report)
```

### 6-Step Pipeline

| Step | What It Does | Example |
|------|-------------|---------|
| 1. ftfy repair | Fixes mangled Unicode encoding | â€™ → ' |
| 2. Invisible strip | Removes zero-width chars, joiners | U+200B, U+200C, U+200D |
| 3. NFKC normalize | Canonical Unicode decomposition | ﬁ → fi |
| 4. Homoglyph fold | Cyrillic → ASCII, smart quotes → ASCII | а (Cyrillic) → a (Latin) |
| 5. Interspacing collapse | Collapses spaced-out text | "l i k e" → "like" |
| 6. Whitespace collapse | Normalizes multiple spaces/newlines | Multiple spaces → single |

### Delta Report

```python
{
    "obfuscation_delta": 0.03,        # Ratio of chars changed
    "invisible_chars": 5,              # Zero-width chars removed
    "homoglyphs": 2,                   # Homoglyphs folded
    "interspacing_spans": 1,           # Spaced-out word sequences
    "whitespace_collapsed": True,
    "ftfy_applied": True,
    "attack_types": ["invisible_char", "homoglyph"]
}
```

---

## Language Gate

**Module:** `llm_detector/language_gate.py`

Fairness protection that caps severity for non-English text where the detection analyzers are not validated.

### Function

```python
check_language_support(text, word_count=None) → dict
```

### Support Levels

| Level | Criteria | Effect on Fusion |
|-------|----------|-----------------|
| **SUPPORTED** | FW coverage ≥ 12% AND <10% non-Latin | No cap |
| **REVIEW** | FW coverage 8–12% OR 10–30% non-Latin | Cap to AMBER |
| **UNSUPPORTED** | FW coverage < 8% OR >30% non-Latin | Cap to YELLOW |

### Return Value
```python
{
    "support_level": "SUPPORTED",
    "function_word_coverage": 0.18,
    "non_latin_ratio": 0.02,
    "reason": "English text with adequate function word coverage"
}
```

### Academic References
Liang et al. (2023), Wang et al. (2023) — fairness in AI-text detection.

---

## Text Utilities

**Module:** `llm_detector/text_utils.py`

Shared utilities consumed by multiple analyzers and infrastructure modules.

### Exports

| Function/Constant | Purpose |
|-------------------|---------|
| `ENGLISH_FUNCTION_WORDS` | frozenset of 50 closed-class English words (the, a, is, of, in, ...) |
| `type_token_ratio(tokens)` | Unique tokens / total tokens |
| `get_sentences(text)` | Sentence splitting (spaCy if available, regex fallback) |
| `get_sentence_spans(text)` | Character-level sentence boundaries: `[(text, start, end), ...]` |

### Sentence Splitting
- **With spaCy (`HAS_SPACY`):** Uses spaCy English sentencizer for accurate splitting
- **Fallback:** Regex-based splitting on sentence-ending punctuation with common abbreviation handling

---

## Reporting

**Module:** `llm_detector/reporting.py`

Serializes pipeline results for batch analysis output with attempter profiling and financial impact analysis.

### Functions

#### `profile_attempters(results, min_submissions=2) → list`
Aggregates by attempter name:
- Flag rate: (RED + AMBER + MIXED) / total submissions
- Primary detection channel for flagged submissions
- Sorted by flag rate descending

#### `channel_pattern_summary(results) → dict`
Breakdown of flagged submissions by detection channel with example task IDs.

#### `financial_impact(results, cost_per_prompt=400.0) → dict`
Calculates waste from flagged submissions:
- Total spend, waste estimate, clean yield
- Projected annual waste (assuming 4 quarterly batches)
- Projected savings at 60% intervention rate

#### `print_attempter_report(results)` / `print_financial_report(results)`
Formatted console output for batch review.

---

## HTML Report Generation

**Module:** `llm_detector/html_report.py`

Renders detection results as standalone HTML files with inline span highlighting.

### Features

- **Span-level highlighting:** Character-level annotation of flagged text regions
- **Signal type colors:** CRITICAL/HIGH (red), MEDIUM (orange), fingerprint (purple), hot_window (red)
- **Channel table:** Score bars for all 4 channels with role badges
- **Fusion box:** Triggering rule explanation with evidence summary
- **Legend:** Color key for all signal types
- **Responsive design:** 900px max-width, self-contained CSS

### Color Scheme

| Signal Type | Color |
|------------|-------|
| CRITICAL/HIGH | `#d32f2f` (red) |
| MEDIUM | `#f57c00` (orange) |
| Pattern/keyword | `#e65100` (dark orange) |
| Fingerprint | `#7b1fa2` (purple) |
| Hot window | `#c62828` (dark red) |

---

## Compatibility & Feature Flags

**Module:** `llm_detector/compat.py`

Centralizes optional dependency detection and lazy model loading.

### Feature Flags

| Flag | Dependency | What It Enables |
|------|-----------|----------------|
| `HAS_TK` | tkinter | Desktop GUI |
| `HAS_SPACY` | spaCy ≥ 3.0 | Accurate sentence tokenization |
| `HAS_FTFY` | ftfy ≥ 6.0 | Unicode encoding repair |
| `HAS_SEMANTIC` | sentence-transformers ≥ 2.0 + sklearn | Semantic resonance, TOCSIN, semantic flow |
| `HAS_PERPLEXITY` | transformers ≥ 4.20 + torch ≥ 2.0 | Perplexity, binoculars, surprisal |
| `HAS_BINOCULARS` | Same as HAS_PERPLEXITY | Contrastive model scoring |
| `HAS_PYPDF` | pypdf ≥ 3.0 | PDF file input |

### Lazy Model Loaders

| Function | What It Loads | Cache |
|----------|-------------|-------|
| `get_nlp()` | spaCy English sentencizer | Global singleton |
| `get_semantic_models()` | all-MiniLM-L6-v2 + centroids | Global singleton |
| `get_perplexity_model(model_id)` | Qwen2.5-0.5B or alternative | Global, swappable |
| `get_binoculars_model()` | DistilGPT-2 as observer model | Global singleton |

### Available Perplexity Models

| Model | ID | Size |
|-------|------|------|
| Qwen2.5-0.5B (default) | `Qwen/Qwen2.5-0.5B` | ~500M params |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` | ~360M params |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` | ~135M params |
| DistilGPT-2 | `distilgpt2` | ~82M params |
| GPT-2 | `gpt2` | ~124M params |

### HuggingFace Token
Loaded from `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` environment variables for gated models.

---

## Shared Constants

**Module:** `llm_detector/_constants.py`

```python
GROUND_TRUTH_LABELS = ['ai', 'human', 'unsure']
STREAMLIT_MIN_VERSION = 'streamlit>=1.20'
```

These are imported by CLI, GUI, and dashboard to ensure consistent label validation and version checking.

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Pipeline Orchestration](pipeline.md) — How infrastructure modules are called
- [Configuration & Dependencies](configuration.md) — Dependency management
