# Detection Analyzers

> Detailed documentation for all 13+ detection analyzers — the core signal extraction layer.

**Module directory:** `llm_detector/analyzers/`

---

## Overview

Each analyzer targets a specific authorship signal. No single analyzer is definitive — signals are combined through [scoring channels](channels.md) and [evidence fusion](fusion.md) to produce a final determination. Every analyzer follows the same contract:

- **Input:** Text string (plus optional configuration parameters)
- **Output:** A result dictionary or tuple containing scores, signal counts, and severity classifications
- **Graceful degradation:** If a required dependency is unavailable, the analyzer returns zero/empty results with an explanatory reason string

---

## Table of Contents

1. [Preamble Detection](#1-preamble-detection)
2. [Fingerprint Detection](#2-fingerprint-detection)
3. [Prompt Signature](#3-prompt-signature)
4. [Voice Dissonance](#4-voice-dissonance)
5. [Instruction Density](#5-instruction-density)
6. [Self-Similarity (NSSI)](#6-self-similarity-nssi)
7. [Semantic Resonance](#7-semantic-resonance)
8. [Continuation API (DNA-GPT)](#8-continuation-api-dna-gpt)
9. [Continuation Local (DNA-GPT Proxy)](#9-continuation-local-dna-gpt-proxy)
10. [Perplexity](#10-perplexity)
11. [Token Cohesiveness (TOCSIN)](#11-token-cohesiveness-tocsin)
12. [Semantic Flow](#12-semantic-flow)
13. [Stylometry](#13-stylometry)
14. [Windowing](#14-windowing)

---

## 1. Preamble Detection

**Module:** `analyzers/preamble.py`
**Channel:** Prompt Structure (critical early-exit signal)
**Min text length:** None

### Purpose
Catches LLM output artifacts that remain when a contributor copy-pastes generated content without cleanup. These are the highest-confidence signals — a ChatGPT acknowledgment like "Sure, here's your task prompt" is near-definitive evidence.

### Function Signature
```python
run_preamble(text) → (score, severity, hits, spans)
```

### Detection Patterns (15 categories)

| Pattern Category | Examples | Severity |
|-----------------|----------|----------|
| Assistant acknowledgments | "got it", "sure thing", "absolutely" | CRITICAL (0.99) |
| Artifact delivery | "here is your", "below is a rewritten" | CRITICAL (0.99) |
| Chain-of-Thought tags | `<think>`, `</think>`, `<reasoning>` | CRITICAL (0.99) |
| First-person creation | "I've created", "I drafted" | CRITICAL (0.99) |
| Style masking | "natural workplace style", "sounds like human" | HIGH (0.75) |
| Editorial meta-commentary | "notes on what I fixed" | HIGH (0.75) |
| CoT reasoning phrases | "let me rethink", "wait, actually" | HIGH (0.75) |
| Copy-paste instructions | "copy-paste", "ready to use" | MEDIUM (0.50) |
| Meta-design language | "failure-inducing", "designed to test" | MEDIUM (0.50) |
| Step numbering | "step 1:", "step 2:" | MEDIUM (0.50) |

### Return Values
- **score** (float): 0.0–0.99 — highest-severity match found
- **severity** (str): "CRITICAL", "HIGH", "MEDIUM", or "NONE"
- **hits** (list): All matched patterns with category labels
- **spans** (list): Character-level positions of each match `{start, end, text, label, type}`

### How It Contributes
Preamble at CRITICAL severity triggers the L0 early-exit path in fusion — the determination is immediately RED at 0.99 confidence with no further analysis required. This is the single most reliable detection signal.

---

## 2. Fingerprint Detection

**Module:** `analyzers/fingerprint.py`
**Channel:** Stylometric (supporting weight)
**Min text length:** None

### Purpose
Detects LLM-preferred vocabulary — words that appeared with dramatically higher frequency in ChatGPT-era academic publications compared to pre-2022 baselines.

### Function Signature
```python
run_fingerprint(text) → (score, hit_count, rate_per_1000)
```

### Vocabulary (32 words)
`delve`, `utilize`, `comprehensive`, `leverage`, `robust`, `paradigm`, `nuanced`, `multifaceted`, `streamline`, `pivotal`, `intricate`, `holistic`, `underscores`, `imperative`, `facilitating`, `commendable`, `noteworthy`, `groundbreaking`, `meticulous`, `invaluable`, `signifies`, `endeavors`, `tapestry`, `realm`, `embark`, `foster`, `showcasing`, and more.

### Scoring
- **Rate:** hits per 1,000 words
- **Score:** `min(rate / 5.0, 1.0)` (normalized to 0.0–1.0)

### Academic Reference
Kobak et al. (2024) "Excess vocabulary" analysis in *Science Advances* — documented the surge of specific words in post-ChatGPT academic papers.

### How It Contributes
Fingerprints are a supporting signal only — they add +0.10 to the stylometric channel score when other signals are already active. On their own, fingerprint words don't trigger a determination because some humans genuinely use these words.

---

## 3. Prompt Signature

**Module:** `analyzers/prompt_signature.py`
**Channel:** Prompt Structure (primary task_prompt signal)
**Min text length:** None

### Purpose
Detects the structural patterns of LLM-generated task prompts: dense constraint frames, must-frame saturation, meta-evaluation design language, and specification completeness.

### Function Signature
```python
run_prompt_signature(text) → dict
```

### Signals

| Signal | Description | Calculation |
|--------|-------------|-------------|
| **CFD** (Constraint Frame Density) | Density of constraint-bearing sentences | constraint_sentences / total_sentences |
| **MFSR** (Must-Frame Saturation Rate) | Sentences with ≥2 constraint frames | multi_frame_sentences / total_sentences |
| **Framing completeness** | Role + deliverable + closing presence | 0–3 scale |
| **Conditional density** | if/when/unless pattern frequency | count per sentence |
| **Meta-design hits** | 12 patterns: rubrics, grading, acceptance criteria | match count |
| **Must rate** | Frequency of "must" keyword | occurrences per sentence |
| **Numbered criteria** | `1.` / `2)` style numbered lists | count of numbered items |
| **Contraction absence** | Lack of contractions (signal of formality) | boolean |

### Lexicon Enhancement
When combined with the [lexicon pack system](lexicon.md), prompt signature scoring is boosted by:
- Constraint packs (obligation, prohibition, recommendation, conditional, cardinality, state)
- Exec-spec packs (gherkin, rubric, acceptance)

Enhanced scoring adds up to +0.45 bonus based on: total constraint score ≥ 0.40 (+0.20), active families ≥ 6 (+0.15), uppercase RFC 2119 hits ≥ 3 (+0.10).

### How It Contributes
The composite score feeds directly into the Prompt Structure channel. Scores ≥ 0.60 → RED, ≥ 0.40 → AMBER, ≥ 0.20 → YELLOW.

---

## 4. Voice Dissonance

**Module:** `analyzers/voice_dissonance.py`
**Channel:** Prompt Structure (primary), Stylometric (supporting)
**Min text length:** None

### Purpose
Detects the contradiction between casual voice markers and technical specification density. Real human contributors tend to write either casually OR formally. LLM-generated content often mixes a forced casual voice with dense technical specifications — a distinctive dissonance.

### Function Signature
```python
run_voice_dissonance(text) → dict
```

### Signals

| Component | Markers | Examples |
|-----------|---------|----------|
| **Voice score** | Casual markers | "hey", "ok so", "gonna", "lol", contractions, em-dashes, lowercase sentence starts |
| **Spec score** | Technical density | camelCase columns, filenames, calculations, tabs, column listings, parentheticals |
| **VSD product** | voice × spec | Multiplicative — high only when BOTH are present |
| **Hedge phrases** | Uncertainty markers | "pretty sure", "i think", "probably" |
| **SSI (Sterile Spec Index)** | Spec without voice | High spec + no voice + no hedges = sterile specification |

### Severity Thresholds

| Condition | Severity | Score |
|-----------|----------|-------|
| VSD ≥ 50 (voice-gated) | RED | 0.90 |
| VSD ≥ 21 (voice-gated) | AMBER | 0.70 |
| VSD ≥ 100 (ungated) | AMBER | 0.70 |
| VSD ≥ 21 (ungated) | YELLOW | 0.45 |
| SSI ≥ 8.0 (spec ≥ 5–7, no voice, no hedges) | AMBER | 0.70 |

### Lexicon Enhancement
Schema packs (schema_json, schema_types, data_fields, tabular) and format packs (format_markup) boost the spec score calculation: `schema_per100 × 2.0 + format_per100 × 1.0`.

### How It Contributes
VSD is a primary signal for task_prompt mode in the Prompt Structure channel and a supporting signal in the Stylometric channel.

---

## 5. Instruction Density

**Module:** `analyzers/instruction_density.py`
**Channel:** Prompt Structure (primary task_prompt signal)
**Min text length:** None

### Purpose
Catches formal-exhaustive LLM output by counting imperative, conditional, and specification language — the hallmarks of an LLM that was asked to "write detailed instructions."

### Function Signature
```python
run_instruction_density(text) → dict
```

### Signals

| Signal | Keywords/Patterns | Count Method |
|--------|------------------|--------------|
| **Imperatives** | must, include, create, load, set, show, use, derive, treat, mark | Per-sentence keyword count |
| **Conditionals** | if, otherwise, when, unless | Per-sentence keyword count |
| **Binary specs** | Yes/No | Exact match count |
| **MISSING handling** | MISSING keyword | Count |
| **Flag count** | FLAG keyword | Count |

### IDI Formula
```
IDI = imperatives + conditionals + binary_specs + missing + flag_count
```

### Severity Thresholds
- **RED:** IDI ≥ 12
- **AMBER:** IDI ≥ 8
- **YELLOW:** IDI < 8 but > 0

### Lexicon Enhancement
Instruction packs (task_verbs, value_domain) boost IDI:
```
enhanced_IDI = legacy_IDI + (task_verb_per100 × weight) + (value_domain_per100 × 2.0)
```
Task verb weight is 1.0 when constraint/schema packs are also active, 0.5 otherwise.

---

## 6. Self-Similarity (NSSI)

**Module:** `analyzers/self_similarity.py`
**Channel:** Stylometric (primary generic_aigt signal)
**Min text length:** ~150 words
**Requires:** `run_l3=True`

### Purpose
The N-Gram Self-Similarity Index detects LLM-generated expository text via convergence of 13 independent formulaic pattern signals. LLM text tends to be self-similar — it reuses the same phrases, structures, and patterns throughout.

### Function Signature
```python
run_self_similarity(text) → dict
```

### 13-Signal Convergence Model

| # | Signal | What It Measures |
|---|--------|-----------------|
| 1 | Formulaic phrase density | 19 academic/discourse patterns |
| 2 | Power adjective density | 30+ high-impact adjectives (comprehensive, robust, etc.) |
| 3 | Discourse scaffolding | Scare quotes, em-dashes, parentheticals, colons |
| 4 | Demonstrative monotony | "this approach", "this issue" repetition |
| 5 | Transition connector density | however, furthermore, consequently |
| 6 | Causal reasoning deficit | Ratio of transitions to causal markers |
| 7 | Sentence-start monotony | "This", "The" opening patterns |
| 8 | Section hierarchy depth | Header/section structure complexity |
| 9 | Sentence length CV | Coefficient of variation of sentence lengths |
| 10 | Zlib compression entropy | High compressibility = repetitive structure |
| 11 | Hapax legomena deficit | Low ratio of unique words (used once) |
| 12 | Structural compression delta | Original vs. word-shuffled compression ratio |
| 13 | Low burstiness | Low variance in sentence length distribution |

### Severity Thresholds
- **RED:** score ≥ 0.70 AND ≥ 7 signals firing
- **AMBER:** score ≥ 0.45 AND ≥ 5 signals firing
- **YELLOW:** score ≥ 0.25 AND ≥ 4 signals firing

### How It Contributes
NSSI is the primary signal for the Stylometric channel in generic_aigt mode. It is the strongest single signal for detecting LLM-generated expository prose.

---

## 7. Semantic Resonance

**Module:** `analyzers/semantic_resonance.py`
**Channel:** Stylometric (supporting)
**Min text length:** ~50 words
**Requires:** `HAS_SEMANTIC` (sentence-transformers)

### Purpose
Measures embedding proximity to AI vs. human archetype centroids. LLM-generated text clusters closer to AI archetypes in the sentence-transformer embedding space.

### Function Signature
```python
run_semantic_resonance(text) → dict
```

### How It Works
1. Text is split into ~200-word chunks
2. Each chunk is embedded using all-MiniLM-L6-v2
3. Cosine similarity is computed against AI and human archetype centroids
4. The semantic delta (mean_ai - mean_human) indicates directionality

### Return Values
- **max_ai_sim**: Highest AI similarity across chunks
- **mean_ai_sim**: Average AI similarity
- **max_human_sim / mean_human_sim**: Human similarity scores
- **semantic_delta**: mean_ai - mean_human (positive = AI-leaning)
- **determination**: AMBER, YELLOW, or GREEN

### Severity Thresholds
- **AMBER:** mean_ai ≥ 0.65 AND delta ≥ 0.15
- **YELLOW:** mean_ai ≥ 0.50 AND delta ≥ 0.08

### Academic Reference
Mitchell et al. (2023) "DetectGPT" — semantic density as an AI detection signal.

---

## 8. Continuation API (DNA-GPT)

**Module:** `analyzers/continuation_api.py`
**Channel:** Continuation (primary, both modes)
**Min text length:** ~80 words
**Requires:** `run_l3=True` + `api_key`

### Purpose
The most robust detection method: truncates the text, asks an LLM to regenerate the missing portion, and measures how much the regeneration overlaps with the original. AI-generated text produces higher overlap because the regeneration converges on similar content.

### Function Signature
```python
run_continuation_api(text, model=None, K=32, ...) → dict
```

### How It Works (DNA-GPT Algorithm)
1. **Truncate** text at a ratio γ (default 0.5 — uses the first half as prefix)
2. **Regenerate** the suffix K times using the LLM API (Anthropic or OpenAI)
3. **Measure overlap** between regenerated suffixes and the original suffix using BScore:
   ```
   BScore = weighted F1 of n-gram overlap (2-gram: 0.25, 3-gram: 0.50, 4-gram: 0.25)
   ```
4. **Average** across K samples for stability

### Multi-Truncation Stability (v0.65)
Runs at three truncation points (γ = 0.3, 0.5, 0.7). Low variance across BScores indicates AI content (regenerations converge regardless of truncation point). High variance suggests human content.

### Severity Thresholds
- **RED:** BScore ≥ 0.20 AND max_bscore ≥ 0.22
- **AMBER:** BScore ≥ 0.12
- **YELLOW:** BScore ≥ 0.08

### Academic Reference
Yang et al. (2024) "DNA-GPT" (ICLR 2024) — divergent continuation for AI text detection.

---

## 9. Continuation Local (DNA-GPT Proxy)

**Module:** `analyzers/continuation_local.py`
**Channel:** Continuation (primary, both modes)
**Min text length:** ~80 words
**Requires:** `run_l3=True` (no API key needed)

### Purpose
Zero-LLM alternative to DNA-GPT that uses a backoff 5-gram language model as a surrogate for LLM regeneration. Trades detection power for zero-cost, zero-latency operation.

### Function Signature
```python
run_continuation_local(text, gamma=0.5, K=32, order=5) → dict
```

### 7 Proxy Features

| Feature | What It Measures | AI Signal |
|---------|-----------------|-----------|
| NCD | Normalized Compression Distance (prefix vs suffix) | Low NCD = similar content |
| Internal overlap | 3-4 gram echo between prefix and suffix | High overlap = AI |
| Repeated n-gram rate | 4-gram monotonicity across text | High repetition = AI |
| Conditional surprisal | Mean token log-probability | Low surprisal = AI |
| TTR | Type-token ratio | Low TTR = AI |
| Surprisal improvement curve | How predictability improves with context | Low improvement = AI |
| NCD matrix variance | Pairwise compression distance across 4 segments | Low variance = AI |

### Composite Score
Weighted combination of all 7 features, normalized to 0.0–1.0.

### Severity Thresholds
- **RED:** composite ≥ 0.60 AND (NCD ≥ 0.4 OR overlap ≥ 0.5)
- **AMBER:** composite ≥ 0.40
- **YELLOW:** composite ≥ 0.25

### Academic References
- Yang et al. (2024) "DNA-GPT" — the continuation framework
- Li et al. (2004) "The Similarity Metric" — NCD theory

---

## 10. Perplexity

**Module:** `analyzers/perplexity.py`
**Channel:** Stylometric (supporting)
**Min text length:** ~50 words
**Requires:** `HAS_PERPLEXITY` (transformers + torch)

### Purpose
Measures how predictable text is to a language model. AI-generated text tends to have lower perplexity (it's more "expected" by the model) and more uniform surprisal patterns.

### Function Signature
```python
run_perplexity(text, model_id=None) → dict
```

### Signals

| Signal | Description | AI Indicator |
|--------|-------------|-------------|
| **Perplexity** | Mean per-token loss (Qwen2.5-0.5B default) | Low PPL = AI |
| **Surprisal variance** (DivEye) | Variance of per-token loss | Low variance = AI |
| **Volatility decay** | first_half_var / second_half_var | High ratio = AI (decaying surprise) |
| **Compression-PPL divergence** | PPL × zlib compression ratio | Compound signal |
| **Per-sentence PPL variance** | Burstiness across sentences | Low variance = AI |
| **Binoculars score** | Performer/observer model ratio | Low ratio = AI |
| **Token losses** | Full per-token loss array | For downstream analysis |

### Available Models
| Model | ID |
|-------|-------|
| Qwen2.5-0.5B (default) | `Qwen/Qwen2.5-0.5B` |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` |
| DistilGPT-2 | `distilgpt2` |
| GPT-2 | `gpt2` |

### Severity Thresholds
- **AMBER:** PPL ≤ 15.0
- **YELLOW:** PPL ≤ 25.0
- **Compound boost:** DivEye (variance < 2.0) + Volatility (decay > 1.5) → severity upgrade

### Academic References
- GLTR (Gehrmann et al. 2019) — token-level prediction visualization
- DetectGPT (Mitchell et al. 2023) — model-based detection
- Hans et al. (2024) "Spotting LLMs With Binoculars" — contrastive model scoring

---

## 11. Token Cohesiveness (TOCSIN)

**Module:** `analyzers/token_cohesiveness.py`
**Channel:** Stylometric (supporting)
**Min text length:** ~40 words
**Requires:** `HAS_SEMANTIC` (sentence-transformers)

### Purpose
Measures semantic stability under random word deletion. AI text is more "cohesive" — deleting random words doesn't change its meaning much because the information is redundant. Human text is more fragile — each word contributes unique meaning.

### Function Signature
```python
run_token_cohesiveness(text, n_copies=10, deletion_rate=0.015, seed=42) → dict
```

### How It Works
1. Embed the original text
2. Repeat 10 times:
   a. Randomly delete 1.5% of words
   b. Embed the modified text
   c. Measure cosine distance from original
3. Report mean cohesiveness (1 - mean_distance) and standard deviation

### Severity Thresholds
- **AMBER:** cohesiveness ≥ 0.020 AND std < 0.010
- **YELLOW:** cohesiveness ≥ 0.012

### Academic Reference
Ma & Wang (EMNLP 2024) "Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness"

---

## 12. Semantic Flow

**Module:** `analyzers/semantic_flow.py`
**Channel:** Stylometric (supporting)
**Min text length:** ~5 sentences
**Requires:** `HAS_SEMANTIC` (sentence-transformers)

### Purpose
Measures inter-sentence embedding similarity variance. LLMs produce uniformly smooth transitions between sentences (low variance); human writing jumps erratically between topics (high variance).

### Function Signature
```python
run_semantic_flow(text, min_sentences=5) → dict
```

### How It Works
1. Split text into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentences
4. Report variance, mean, and standard deviation of similarities

### Return Values
- **flow_variance**: Key signal — low variance = AI
- **flow_mean**: Average inter-sentence similarity
- **similarities**: List of per-pair similarities
- **determination**: AMBER, YELLOW, or GREEN

### Severity Thresholds
- **AMBER:** variance < 0.008 AND mean > 0.40
- **YELLOW:** variance < 0.015 AND mean > 0.35

---

## 13. Stylometry

**Module:** `analyzers/stylometry.py`
**Channel:** Utility (feature extraction for other analyzers)
**Min text length:** None

### Purpose
Extracts topic-scrubbed stylometric features. Content words are masked to isolate writing style from topic, reducing domain leakage.

### Function Signature
```python
extract_stylometric_features(text, masked_text=None) → dict
```

### Features Extracted

| Feature | Description |
|---------|-------------|
| **MATTR** | Moving-Average Type-Token Ratio (50-word windows) |
| **Function word ratio** | 35+ English function words / total words |
| **Punctuation bigrams** | Top character pairs (e.g., ". ", ", ") |
| **Sentence length dispersion** | std(sentence_lengths) / mean(sentence_lengths) |
| **Type-token ratio** | Unique tokens / total tokens |
| **Average word length** | Mean character count per word |
| **Short word ratio** | Words ≤ 3 characters / total words |
| **Character 4-grams** | Top 50 most frequent 4-character sequences |

### How It Contributes
Stylometry is a utility module — it provides features consumed by other analyzers and channels rather than producing its own severity determination.

---

## 14. Windowing

**Module:** `analyzers/windowing.py`
**Channel:** Windowed (primary generic_aigt signal)
**Min text length:** ~5 sentences

### Purpose
Detects mixed human+AI content by analyzing text in overlapping sentence windows. Identifies "hot spans" where AI-like patterns concentrate, and uses CUSUM changepoint detection to find human→AI transition boundaries.

### Function Signature
```python
score_windows(text, window_size=5, stride=2) → dict
```

### Signals

| Signal | Description | AI Indicator |
|--------|-------------|-------------|
| **Per-window scores** | NSSI-like signals per window | High-scoring windows = AI sections |
| **Hot span length** | Consecutive high-scoring windows | Long hot spans = AI blocks |
| **FW trajectory CV** | Function word ratio variation | Low CV = uniform AI |
| **Compression profile** | Per-window zlib ratio | Uniform compression = AI |
| **Changepoint (CUSUM)** | Content shift detection | Transition boundary found |
| **Mixed signal** | Variance + max + mean pattern | High variance + high max = mixed content |

### Severity Thresholds
- **RED:** max_window ≥ 0.60 AND hot_span ≥ 3
- **AMBER:** max_window ≥ 0.45 AND hot_span ≥ 2
- **YELLOW:** max_window ≥ 0.30

### Academic Reference
Wang et al. (2024) M4GT-Bench — mixed content detection as a separate evaluation task.

---

## Analyzer Summary Table

| Analyzer | Channel | Mode | Primary/Supporting | Key Signal | Dependency |
|----------|---------|------|-------------------|------------|------------|
| Preamble | Prompt Structure | task_prompt | Critical (early exit) | LLM output artifacts | None |
| Fingerprint | Stylometric | generic_aigt | Supporting | Excess vocabulary | None |
| Prompt Signature | Prompt Structure | task_prompt | Primary | Constraint density | None + Lexicon |
| Voice Dissonance | Prompt Structure + Stylometric | task_prompt | Primary (struct) | Voice-spec mismatch | None + Lexicon |
| Instruction Density | Prompt Structure | task_prompt | Primary | Imperative density | None + Lexicon |
| Self-Similarity | Stylometric | generic_aigt | Primary | Formulaic convergence | None |
| Semantic Resonance | Stylometric | generic_aigt | Supporting | Embedding proximity | sentence-transformers |
| Continuation (API) | Continuation | both | Primary | Regeneration overlap | API key |
| Continuation (Local) | Continuation | both | Primary | N-gram proxy features | None |
| Perplexity | Stylometric | generic_aigt | Supporting | Low perplexity | transformers + torch |
| Token Cohesiveness | Stylometric | generic_aigt | Supporting | Deletion stability | sentence-transformers |
| Semantic Flow | Stylometric | generic_aigt | Supporting | Transition uniformity | sentence-transformers |
| Stylometry | Utility | N/A | Feature extraction | Style features | None |
| Windowing | Windowed | generic_aigt | Primary | Hot spans + changepoint | None |

---

## Related Documentation

- [Architecture Overview](architecture.md) — System-level design
- [Scoring Channels](channels.md) — How analyzers are aggregated
- [Evidence Fusion](fusion.md) — How channel results become determinations
- [Lexicon Pack System](lexicon.md) — Vocabulary families that enhance analyzers
