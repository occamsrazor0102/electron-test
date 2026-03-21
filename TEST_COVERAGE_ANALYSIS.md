# Test Coverage Analysis

**Overall coverage:** 38% (4,204 statements missed out of 6,829)
**Tests passing:** 152/152

---

## Per-Module Coverage Breakdown

| Module | Stmts | Miss | Coverage | Status |
|--------|-------|------|----------|--------|
| `__init__.py` | 12 | 0 | **100%** | Good |
| `analyzers/__init__.py` | 12 | 0 | **100%** | Good |
| `analyzers/fingerprint.py` | 15 | 0 | **100%** | Good |
| `analyzers/instruction_density.py` | 14 | 0 | **100%** | Good |
| `analyzers/preamble.py` | 26 | 0 | **100%** | Good |
| `analyzers/prompt_signature.py` | 53 | 1 | **98%** | Good |
| `analyzers/stylometry.py` | 52 | 1 | **98%** | Good |
| `analyzers/voice_dissonance.py` | 40 | 1 | **98%** | Good |
| `calibration.py` | 82 | 4 | **95%** | Good |
| `channels/__init__.py` | 21 | 1 | **95%** | Good |
| `analyzers/continuation_local.py` | 225 | 19 | **92%** | Good |
| `pipeline.py` | 84 | 7 | **92%** | Good |
| `lexicon/integration.py` | 127 | 13 | **90%** | Good |
| `analyzers/self_similarity.py` | 140 | 15 | **89%** | OK |
| `analyzers/windowing.py` | 140 | 15 | **89%** | OK |
| `channels/continuation.py` | 32 | 4 | **88%** | OK |
| `normalize.py` | 60 | 10 | **83%** | OK |
| `similarity.py` | 263 | 64 | **76%** | Needs work |
| `channels/windowed.py` | 28 | 7 | **75%** | Needs work |
| `lexicon/packs.py` | 176 | 48 | **73%** | Needs work |
| `fusion.py` | 121 | 33 | **73%** | Needs work |
| `text_utils.py` | 27 | 8 | **70%** | Needs work |
| `language_gate.py` | 29 | 9 | **69%** | Needs work |
| `reporting.py` | 103 | 39 | **62%** | Needs work |
| `html_report.py` | 104 | 44 | **58%** | Needs work |
| `io.py` | 115 | 59 | **49%** | Poor |
| `channels/stylometric.py` | 109 | 59 | **46%** | Poor |
| `channels/prompt_structure.py` | 76 | 44 | **42%** | Poor |
| `compat.py` | 121 | 70 | **42%** | Poor |
| `memory.py` | 941 | 542 | **42%** | Poor |
| `cli.py` | 778 | 505 | **35%** | Poor |
| `analyzers/continuation_api.py` | 269 | 190 | **29%** | Poor |
| `baselines.py` | 149 | 126 | **15%** | Critical |
| `analyzers/token_cohesiveness.py` | 41 | 35 | **15%** | Critical |
| `analyzers/semantic_resonance.py` | 38 | 33 | **13%** | Critical |
| `analyzers/perplexity.py` | 108 | 101 | **6%** | Critical |
| `__main__.py` | 2 | 2 | **0%** | No tests |
| `dashboard.py` | 764 | 764 | **0%** | No tests |
| `gui.py` | 1331 | 1331 | **0%** | No tests |

---

## Priority Areas for Test Improvement

### Priority 1 — Channel Scoring Layer (42-46% coverage)

**Files:** `channels/prompt_structure.py`, `channels/stylometric.py`

These are the core scoring channels that combine analyzer signals into final determinations. They contain complex threshold logic with multiple severity levels (RED/AMBER/YELLOW/GREEN) and boosting/capping mechanics that directly control the output the user sees.

**What to test:**
- Each severity path (CRITICAL preamble override, RED/AMBER/YELLOW thresholds)
- Voice-gated vs ungated branches in `prompt_structure`
- Supporting signal stacking and score capping at 1.0 in `stylometric`
- Boundary values at each threshold (e.g., prompt_sig scores of 0.19, 0.20, 0.21)
- `data_sufficient = False` when no sub-signals are populated
- Confidence min/max capping logic

**Why it matters:** Bugs here silently shift every determination the pipeline produces. These are pure-function modules with no I/O — easy to unit test.

---

### Priority 2 — Baselines Module (15% coverage)

**File:** `baselines.py` (149 stmts, 126 missed)

Only `derive_attack_type()` has implicit coverage via pipeline tests. The two main functions — `collect_baselines()` and `analyze_baselines()` — are completely untested.

**What to test:**
- `collect_baselines`: appending to JSONL, timestamp/version metadata, length binning
- `analyze_baselines`: percentile tables, flag rate computation, ROC analysis at FPR thresholds
- Edge cases: empty datasets, no labeled data, stratum disparity detection
- `derive_attack_type`: obfuscation delta threshold (0.02), combined vs single attack types

**Why it matters:** This module produces evaluation metrics (ROC, flag rates) that inform whether the system is working correctly. Silent bugs here undermine trust in the entire pipeline.

---

### Priority 3 — I/O Module (49% coverage)

**File:** `io.py` (115 stmts, 59 missed)

Column resolution (positional specs like "A"/"1", fuzzy matching, substring matching) and sheet detection logic are untested. Only `load_pdf` has coverage.

**What to test:**
- `load_xlsx` / `load_csv` with positional column specs (A-Z, 1-26)
- Fuzzy/substring column header matching (case-insensitive)
- Sheet auto-detection priority (FullTaskX > Full Task Connected > Claim Sheet > etc.)
- Missing columns (optional occupation/attempter/stage)
- Task ID truncation at 20 chars
- Empty files or missing prompt column (error handling)

**Why it matters:** This is the system's entry point — if column resolution silently picks the wrong column, every downstream result is wrong.

---

### Priority 4 — Memory Store (42% coverage, 941 statements)

**File:** `memory.py` (941 stmts, 542 missed)

The largest module in the codebase. While basic CRUD and fingerprint operations are tested, the ML tooling (Shadow Model, Lexicon Discovery, Semantic Centroid Rebuilder) and attempter risk profiling are largely untested.

**What to test:**
- Shadow model training and prediction
- Lexicon discovery (Monroe et al. 2008 log-odds)
- Semantic centroid rebuilding
- Attempter risk profiling and profile updates
- Cross-batch similarity with persistence
- Batch summary statistics
- Edge cases: empty stores, single-entry stores, corrupted JSON

**Why it matters:** The memory system accumulates state across batches. Untested accumulation logic can produce silent drift over time.

---

### Priority 5 — Analyzers with External Dependencies (6-15% coverage)

**Files:**
- `analyzers/perplexity.py` (6% — requires `transformers`/`torch`)
- `analyzers/semantic_resonance.py` (13% — requires `sentence-transformers`)
- `analyzers/token_cohesiveness.py` (15% — requires `sentence-transformers`)

These analyzers gate on optional NLP dependencies (`HAS_PERPLEXITY`, `HAS_SEMANTIC`), so they need tests that mock or conditionally skip.

**What to test:**
- Graceful degradation when dependencies are absent (return safe defaults)
- Core logic with mocked models (mock embedder/tokenizer returns)
- Threshold boundary tests for determination logic
- Short-text guards (e.g., `< 30 words` paths)

**Why it matters:** These are high-value detection signals. Without tests, regressions in threshold logic go unnoticed since CI doesn't install the heavy NLP stack.

---

### Priority 6 — CLI Module (35% coverage)

**File:** `cli.py` (778 stmts, 505 missed)

The CLI is the primary user interface. Argument parsing, file dispatch, and output formatting are only partially covered.

**What to test:**
- Argument parsing for all subcommands and flag combinations
- File type dispatch (xlsx, csv, pdf, stdin)
- Output format selection (text, JSON, HTML)
- Error handling (missing files, invalid formats)
- `--batch` mode orchestration
- Memory store integration flags

**Why it matters:** CLI bugs produce confusing user-facing errors. Argument parsing is easy to test with `argparse` or click testing utilities.

---

### Priority 7 — Fusion Module (73% coverage)

**File:** `fusion.py` (121 stmts, 33 missed)

Evidence fusion combines channel results into a final verdict. Missed lines include the conformal calibration path, language gate severity capping, and the multi-channel conflict resolution logic.

**What to test:**
- Calibration path (when `calibrator` is provided)
- Language gate capping (UNSUPPORTED/REVIEW texts)
- Channel conflict scenarios (e.g., continuation says RED, stylometric says GREEN)
- Final severity promotion/demotion rules

**Why it matters:** Fusion is the final decision point. Bugs here override correct channel-level analysis.

---

### Priority 8 — Text Utilities and Language Gate (69-70% coverage)

**Files:** `text_utils.py` (70%), `language_gate.py` (69%)

**What to test for `text_utils.py`:**
- `get_sentences` with spaCy unavailable (regex fallback path)
- `get_sentence_spans` for texts with no punctuation
- `type_token_ratio` with empty token lists

**What to test for `language_gate.py`:**
- Boundary values for function-word coverage (8%, 12%)
- Non-Latin ratio thresholds (10%, 30%)
- Mixed-script content (Latin + CJK, Cyrillic, Arabic)
- Short text fallback (< 30 words)

---

### Lower Priority — GUI/Dashboard (0% coverage)

**Files:** `gui.py` (1331 stmts), `dashboard.py` (764 stmts)

These are UI layers (Tkinter desktop GUI and Streamlit web dashboard). They represent 2,095 lines with 0% coverage. Full UI testing is impractical, but key logic can be extracted and tested.

**Recommendation:** Extract any non-UI logic (data transformation, state management) into testable helper functions. Smoke-test that the modules import without errors.

---

## Recommendations Summary

| Priority | Area | Current | Target | Effort |
|----------|------|---------|--------|--------|
| P1 | Channel scoring (prompt_structure, stylometric) | 42-46% | 90%+ | Low |
| P2 | Baselines | 15% | 80%+ | Medium |
| P3 | I/O (load_xlsx, load_csv) | 49% | 85%+ | Medium |
| P4 | Memory (ML tools, risk profiles) | 42% | 70%+ | High |
| P5 | NLP-dependent analyzers | 6-15% | 60%+ | Medium |
| P6 | CLI | 35% | 65%+ | Medium |
| P7 | Fusion (calibration, language gate paths) | 73% | 90%+ | Low |
| P8 | text_utils, language_gate | 69-70% | 90%+ | Low |

**Quick wins (low effort, high impact):** P1, P7, and P8 are pure-function modules with no I/O dependencies — they can be brought to 90%+ coverage with straightforward parametrized unit tests.

**Biggest risk:** P1 (channel scoring) and P2 (baselines) — bugs in these modules directly affect detection accuracy and evaluation metrics without any visible error.
