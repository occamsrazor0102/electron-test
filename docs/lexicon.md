# Lexicon Pack System

> 16 externalized vocabulary families that enhance detection analyzers with domain-specific pattern matching.

**Module directory:** `llm_detector/lexicon/`

---

## Overview

The lexicon pack system provides externalized, versioned vocabulary families that feed into specific detection analyzers. Rather than hardcoding keyword lists inside analyzers, patterns are organized into reusable "packs" with independent weights and caps. This design allows:

- **Adding new vocabulary** without modifying analyzer code
- **Per-pack tuning** — each pack has its own weight and maximum contribution
- **Category organization** — packs are grouped by semantic function
- **Span-level annotation** — every match produces character-level spans for visualization

---

## Architecture

```
lexicon/
├── packs.py         # 16 LexiconPack dataclass definitions (~43KB)
└── integration.py   # Enhanced analyzer wrappers (~9KB)
```

### LexiconPack Structure

Each pack is a frozen dataclass with:

```python
@dataclass(frozen=True)
class LexiconPack:
    name: str              # Pack identifier (e.g., 'obligation')
    category: str          # Semantic category (e.g., 'constraint')
    target_layer: str      # Which analyzer consumes it (e.g., 'prompt_signature')
    mode: str              # Mode eligibility: 'task_prompt', 'generic_aigt', or 'both'
    patterns: tuple        # Regex patterns with per-hit weights
    keywords: frozenset    # Literal word-boundary matches (case-insensitive)
    uppercase_keywords: frozenset  # Case-sensitive RFC 2119 normative keywords
    family_weight: float   # Global multiplier for this pack's contribution
    family_cap: float      # Maximum score this pack can contribute (0.0–1.0)
    version: str           # Pack version
    source: str            # Reference/source documentation
```

---

## All 16 Lexicon Packs

### Constraint Packs (6 packs → Prompt Signature analyzer)

These detect RFC 2119 / specification-style constraint language commonly found in LLM-generated task prompts.

#### 1. `obligation`
- **Category:** constraint
- **Purpose:** RFC 2119 MUST/REQUIRED/SHALL (normative obligation)
- **Uppercase keywords:** MUST, REQUIRED, SHALL
- **Patterns:** must, required, mandatory, obligatory
- **Target:** prompt_signature

#### 2. `prohibition`
- **Category:** constraint
- **Purpose:** RFC 2119 MUST NOT/SHALL NOT, task-prompt negation
- **Uppercase keywords:** MUST NOT, SHALL NOT
- **Patterns:** never, prohibited, avoid, forbidden, do not
- **Target:** prompt_signature

#### 3. `recommendation`
- **Category:** constraint
- **Purpose:** RFC 2119 SHOULD/RECOMMENDED/MAY/OPTIONAL
- **Uppercase keywords:** SHOULD, RECOMMENDED, MAY, OPTIONAL
- **Patterns:** ideally, preferably, suggested, advisable
- **Target:** prompt_signature

#### 4. `conditional`
- **Category:** constraint
- **Purpose:** EARS event/state/conditional scaffolds
- **Patterns:** if, when, while, unless, given, provided that, in the event of
- **Keywords:** conditional, prerequisite, trigger
- **Target:** prompt_signature

#### 5. `cardinality`
- **Category:** constraint
- **Purpose:** Quantification operators
- **Patterns:** exactly N, at least N, at most N, between X and Y, each, all, one of
- **Keywords:** minimum, maximum, exactly
- **Target:** prompt_signature

#### 6. `state`
- **Category:** constraint
- **Purpose:** Lifecycle and state-transition operators
- **Patterns:** initial state, final state, on success, on error, transitions to
- **Keywords:** initialized, finalized, pending, completed, active, expired
- **Target:** prompt_signature

---

### Schema Packs (4 packs → Voice Dissonance analyzer)

These detect data structure and API vocabulary that amplifies the specification density component of voice dissonance.

#### 7. `schema_json`
- **Category:** schema
- **Purpose:** JSON Schema / OpenAPI vocabulary
- **Keywords:** schema, endpoint, payload
- **Patterns:** json, yaml, http, api, REST, GraphQL
- **Target:** voice_dissonance

#### 8. `schema_types`
- **Category:** schema
- **Purpose:** Type system and property control
- **Keywords:** nullable, properties
- **Patterns:** type:, required:, enum, allOf, oneOf, anyOf, $ref
- **Target:** voice_dissonance

#### 9. `data_fields`
- **Category:** schema
- **Purpose:** Generic data structure vocabulary
- **Keywords:** field, key, value, record, attribute
- **Patterns:** column, primary key, foreign key, index
- **Target:** voice_dissonance

#### 10. `tabular`
- **Category:** schema
- **Purpose:** Spreadsheet/CSV vocabulary (RFC 4180)
- **Keywords:** delimiter, worksheet, spreadsheet, tabular
- **Patterns:** csv, header row, vlookup, pivot table
- **Target:** voice_dissonance

---

### Exec-Spec Packs (3 packs → Prompt Signature analyzer)

These detect structured specification formats commonly used by LLMs when generating task prompts.

#### 11. `gherkin`
- **Category:** exec_spec
- **Purpose:** Gherkin BDD (Behavior-Driven Development) keywords
- **Patterns:** Feature:, Scenario:, Given, When, Then, And, But, Examples:
- **Target:** prompt_signature

#### 12. `rubric`
- **Category:** exec_spec
- **Purpose:** Evaluation and grading vocabulary
- **Keywords:** rubric, checklist, grader, verification
- **Patterns:** scoring, test case, expected output, pass/fail criteria
- **Target:** prompt_signature

#### 13. `acceptance`
- **Category:** exec_spec
- **Purpose:** Acceptance criteria and definition-of-done
- **Keywords:** deliverable, milestone, criterion
- **Patterns:** acceptance criteria, exit criteria, user story, definition of done
- **Target:** prompt_signature

---

### Instruction Packs (2 packs → Instruction Density analyzer)

These detect action verbs and value-domain operators that indicate formal instruction generation.

#### 14. `task_verbs`
- **Category:** instruction
- **Purpose:** Bloom's Taxonomy action verbs
- **Keywords:** classify, identify, extract, label, compare, evaluate, validate, generate, transform, derive (20 verbs)
- **Patterns:** higher-order cognitive action patterns
- **Target:** instruction_density

#### 15. `value_domain`
- **Category:** instruction
- **Purpose:** Value-domain operators (null handling, defaults, allowed values)
- **Keywords:** null, fallback, default, placeholder, sentinel
- **Patterns:** leave blank, valid values, allowed range, out of range
- **Target:** instruction_density

---

### Format Packs (1 pack → Voice Dissonance analyzer)

#### 16. `format_markup`
- **Category:** format
- **Purpose:** Format and markup sublexicon
- **Patterns:** markdown syntax, code blocks, indentation markers, formatting directives
- **Target:** voice_dissonance

---

## Pack Summary Table

| # | Pack | Category | Target Analyzer | Purpose |
|---|------|----------|----------------|---------|
| 1 | obligation | constraint | prompt_signature | MUST/REQUIRED/SHALL |
| 2 | prohibition | constraint | prompt_signature | MUST NOT/SHALL NOT |
| 3 | recommendation | constraint | prompt_signature | SHOULD/MAY/OPTIONAL |
| 4 | conditional | constraint | prompt_signature | if/when/unless scaffolds |
| 5 | cardinality | constraint | prompt_signature | exactly/at least/at most |
| 6 | state | constraint | prompt_signature | Lifecycle operators |
| 7 | schema_json | schema | voice_dissonance | JSON Schema/OpenAPI |
| 8 | schema_types | schema | voice_dissonance | Type system vocabulary |
| 9 | data_fields | schema | voice_dissonance | Data structure vocabulary |
| 10 | tabular | schema | voice_dissonance | Spreadsheet/CSV vocabulary |
| 11 | gherkin | exec_spec | prompt_signature | BDD Given/When/Then |
| 12 | rubric | exec_spec | prompt_signature | Evaluation/grading |
| 13 | acceptance | exec_spec | prompt_signature | Acceptance criteria |
| 14 | task_verbs | instruction | instruction_density | Bloom's Taxonomy verbs |
| 15 | value_domain | instruction | instruction_density | Null/default/allowed values |
| 16 | format_markup | format | voice_dissonance | Markdown/code formatting |

---

## Integration Layer

**Module:** `llm_detector/lexicon/integration.py`

The integration layer provides enhanced analyzer wrappers that combine base analyzer logic with lexicon pack scoring.

### Core Scoring Function

```python
score_packs(text, pack_names, n_sents) → dict[str, PackScore]
```

Each `PackScore` contains:
- **raw_hits:** Total pattern/keyword matches
- **capped_score:** Score after applying family_cap
- **weighted_hits:** Hits × family_weight
- **spans:** Character-level positions of each match `{start, end, text, pack, pattern}`

### Enhanced Analyzer Wrappers

#### `run_prompt_signature_enhanced(text)`
Wraps the base prompt signature analyzer with constraint and exec-spec pack scoring:
- Scores constraint packs: obligation, prohibition, recommendation, conditional, cardinality, state
- Scores exec-spec packs: gherkin, rubric, acceptance
- **Boost rules:**
  - Total constraint score ≥ 0.40 → +0.20 to composite
  - Active families ≥ 6 → +0.15 to composite
  - Uppercase hits ≥ 3 → +0.10 to composite
- Returns enhanced result with: legacy_composite, pack_boost, pack_details, pack_spans

#### `run_voice_dissonance_enhanced(text)`
Wraps the base voice dissonance analyzer with schema and format pack scoring:
- Scores schema packs: schema_json, schema_types, data_fields, tabular
- Scores format packs: format_markup
- **Enhancement:** `enhanced_spec = legacy_spec + schema_per100 × 2.0 + format_per100 × 1.0`
- Recalculates VSD with enhanced spec score
- Returns enhanced result with: legacy_vsd, enhanced_vsd, pack_details, pack_spans

#### `run_instruction_density_enhanced(text, constraint_active, schema_active)`
Wraps the base instruction density analyzer with instruction pack scoring:
- Scores instruction packs: task_verbs, value_domain
- **Task verb weight:** 1.0 if constraint/schema packs are active, 0.5 otherwise
- **Enhancement:** `enhanced_IDI = legacy_IDI + tv_per100 × weight + vd_per100 × 2.0`
- Returns enhanced result with: legacy_idi, enhanced_idi, pack_details, pack_spans

---

## How Packs Feed into the Pipeline

```
Text arrives at pipeline.analyze_prompt()
    │
    ├─→ run_prompt_signature_enhanced()
    │     ├── Base prompt_signature.py
    │     └── Lexicon: obligation, prohibition, recommendation,
    │                  conditional, cardinality, state,
    │                  gherkin, rubric, acceptance
    │
    ├─→ run_voice_dissonance_enhanced()
    │     ├── Base voice_dissonance.py
    │     └── Lexicon: schema_json, schema_types, data_fields,
    │                  tabular, format_markup
    │
    └─→ run_instruction_density_enhanced()
          ├── Base instruction_density.py
          └── Lexicon: task_verbs, value_domain
```

Pack spans from all three enhanced analyzers are collected into the pipeline's `detection_spans` list, enabling character-level visualization of every matched vocabulary pattern.

---

## Related Documentation

- [Detection Analyzers](analyzers.md) — Analyzer details (consumers of lexicon packs)
- [Pipeline Orchestration](pipeline.md) — How enhanced wrappers are called
- [Architecture Overview](architecture.md) — System-level design
