# Test Coverage Summary

**Overall Coverage: 39%** (4,197 statements missed out of 6,829)
**Tests Passing: 152/152** ✓

---

## Quick Reference: Priority Areas for Improvement

### 🔴 Critical (Immediate Action Required)

These modules are core to the detection engine and have significant coverage gaps:

| Module | Current | Target | Impact |
|--------|---------|--------|--------|
| `channels/prompt_structure.py` | 42% | 85% | Primary scoring channel - determines RED/AMBER/YELLOW |
| `channels/stylometric.py` | 46% | 85% | Primary scoring channel - NSSI, semantic, perplexity |
| `fusion.py` | 73% | 95% | Final decision maker - combines all channels |
| `io.py` | 49% | 85% | Entry point for batch analysis (xlsx/csv/pdf) |
| `baselines.py` | 15% | 70% | Calibration data collection and analysis |

**Immediate Effort:** 1 sprint (~2 weeks)
**Expected Gain:** +30 percentage points overall coverage

---

### 🟡 Important (Medium Priority)

These modules have moderate gaps and should be addressed soon:

| Module | Current | Target | Notes |
|--------|---------|--------|-------|
| `memory.py` | 42% | 65% | Attempter risk profiling, ground truth tracking |
| `cli.py` | 35% | 60% | User interface - argument parsing, batch processing |
| `analyzers/perplexity.py` | 6% | 50% | ML-dependent - requires mocking strategy |
| `analyzers/semantic_resonance.py` | 13% | 50% | ML-dependent - requires mocking strategy |
| `analyzers/token_cohesiveness.py` | 15% | 50% | ML-dependent - requires mocking strategy |

**Effort:** 2-3 sprints (~4-6 weeks)
**Expected Gain:** +20 percentage points overall coverage

---

### 🟢 Long-term (Lower Priority)

| Module | Current | Target | Notes |
|--------|---------|--------|-------|
| `dashboard.py` | 0% | 40% | GUI module - requires UI testing framework |
| `gui.py` | 0% | 40% | GUI module - requires UI testing framework |
| `html_report.py` | 58% | 75% | Report generation - mostly covered |

---

## Top 5 Specific Test Gaps

### 1. **Prompt Structure Thresholds (lines 33-105)**
Missing tests for:
- Prompt signature thresholds: 0.20 (YELLOW), 0.40 (AMBER), 0.60 (RED)
- Voice dissonance gated/ungated paths: 21, 50, 100 thresholds
- Instruction density: 8 (AMBER), 12 (RED)
- SSI (Sterile Specification Index) triggering conditions

**Why it matters:** These thresholds directly determine if text is flagged as AI-generated.

---

### 2. **Fusion Edge Cases (lines 85-199)**
Missing tests for:
- Language gate severity caps (UNSUPPORTED → YELLOW, REVIEW → AMBER)
- Channel ablation (disabled channels excluded from fusion)
- Multi-channel convergence (2 AMBER → RED)
- MIXED determination from windowed variance
- Short-text relaxation rules

**Why it matters:** Fusion is the final decision point - bugs here affect every detection.

---

### 3. **I/O Loaders (lines 34-195)**
Missing tests for:
- XLSX default sheet detection (FullTaskX, etc.)
- Positional column references (A, B, C or 1, 2, 3)
- Fuzzy column matching (substring search)
- Short prompt filtering (<50 chars)
- CSV/XLSX error handling

**Why it matters:** Zero coverage means batch file loading is completely untested.

---

### 4. **Baselines Collection & Analysis (lines 64-221)**
Missing tests for:
- JSONL writing and appending
- Percentile computation
- TPR@FPR calculation
- Stratified flag rates by domain × length
- Ground truth labeling

**Why it matters:** Calibration relies on this - inaccurate baselines → false positives/negatives.

---

### 5. **ML Analyzer Mocking (perplexity, semantic, tocsin)**
Missing tests for:
- Surprisal variance calculation
- Semantic resonance delta
- Token cohesiveness fragility
- All core scoring logic

**Why it matters:** 6-15% coverage means ML features are essentially untested.

---

## Testing Infrastructure Gaps

### Missing Configuration
- ❌ No `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml`
- ❌ No coverage enforcement in CI/CD (`--cov-fail-under=60`)
- ❌ No test markers for slow/GPU/network tests
- ❌ No coverage reporting (Codecov or similar)

### Test Pattern Issues
- Uses custom `check(label, condition, detail)` instead of standard `assert`
- No pytest plugins (pytest-cov, pytest-xdist, pytest-timeout)
- Limited edge-case fixtures (short text, non-English, obfuscated)

---

## Recommended Action Plan

### Sprint 1 (Immediate - 2 weeks)
1. Add 50+ tests for `channels/prompt_structure.py` → 42% to 85%
2. Add 40+ tests for `channels/stylometric.py` → 46% to 85%
3. Add 20+ tests for `fusion.py` edge cases → 73% to 95%
4. Add 30+ tests for `io.py` loaders → 49% to 85%
5. Configure pytest with 60% coverage threshold

**Outcome:** 39% → 55% overall coverage (+16pp)

### Sprint 2-3 (Medium-term - 4-6 weeks)
6. Add 30+ tests for `baselines.py` → 15% to 70%
7. Add 25+ tests for `memory.py` → 42% to 65%
8. Mock ML analyzers and add 40+ tests → 6-15% to 50%
9. Add edge-case fixtures (short, non-English, obfuscated, mixed)
10. Set up CI/CD coverage reporting

**Outcome:** 55% → 70% overall coverage (+15pp)

### Sprint 4+ (Long-term)
11. GUI/Dashboard testing (requires Playwright or Selenium)
12. CLI integration tests
13. Performance benchmarking
14. Mutation testing for test quality

**Outcome:** 70% → 75%+ overall coverage (+5pp)

---

## Key Metrics

| Metric | Current | Target (Sprint 1) | Target (Sprint 3) | Final Goal |
|--------|---------|-------------------|-------------------|------------|
| Overall Coverage | 39% | 55% | 70% | 75%+ |
| Critical Modules | 42-73% | 85%+ | 85%+ | 90%+ |
| ML Modules | 6-15% | 30% | 50% | 60% |
| Test Count | 152 | 250+ | 400+ | 500+ |

---

## Getting Started

**Step 1:** Read the full analysis: `TEST_COVERAGE_ANALYSIS.md`

**Step 2:** Pick a module from the Critical list (e.g., `channels/prompt_structure.py`)

**Step 3:** Copy the test cases from the detailed recommendations

**Step 4:** Run tests with coverage:
```bash
pytest --cov=llm_detector --cov-report=term-missing -v
```

**Step 5:** Iterate until module reaches target coverage

---

**Last Updated:** 2026-03-10
**Analysis Tool:** pytest-cov 5.x
**Python Version:** 3.12.3
