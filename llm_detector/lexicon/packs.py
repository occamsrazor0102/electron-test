#!/usr/bin/env python3
"""
BEET Lexicon Packs v1.0
═══════════════════════
Externalized, versioned detection vocabulary organized into typed families
with independent weights and caps per family.

Design rationale (from roadmap):
  "Flat word bags are where detectors go to die."
  Each pack is a named family with:
    - A semantic category (obligation, schema, gherkin, etc.)
    - Patterns (regex or literal keywords)
    - A weight (contribution strength per hit)
    - A cap (maximum contribution from this family alone)
    - A target layer (which pipeline layer consumes it)
    - Mode eligibility (task_prompt, generic_aigt, or both)

Usage:
    from lexicon_packs import PACK_REGISTRY, get_packs_for_layer, score_packs

    # Get all packs feeding prompt_signature layer
    ps_packs = get_packs_for_layer('prompt_signature')

    # Score text against packs
    results = score_packs(text, ps_packs)
    # → {'obligation': {'hits': 5, 'score': 0.35, 'capped': 0.30, 'matches': [...]}, ...}
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, FrozenSet

__version__ = '1.0.0'
__pack_date__ = '2026-03-04'


# ══════════════════════════════════════════════════════════════════════════════
# PACK DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LexiconPack:
    """A single vocabulary family with scoring parameters.

    Attributes:
        name:       Unique identifier (e.g., 'obligation', 'schema_json')
        category:   Semantic grouping ('constraint', 'schema', 'exec_spec',
                    'instruction', 'format', 'discourse')
        target_layer: Which pipeline layer consumes this ('prompt_signature',
                      'voice_dissonance', 'instruction_density', 'self_similarity', 'multi')
        mode:       'task_prompt', 'generic_aigt', or 'both'
        patterns:   List of (regex_string, per_hit_weight) tuples.
                    Regexes are compiled with re.IGNORECASE by default.
        keywords:   Frozenset of literal lowercase keywords (fast path).
                    Checked via word-boundary match.
        uppercase_keywords: Frozenset of UPPERCASE keywords where case matters
                    (e.g., BCP 14 normative forms). Matched case-sensitively.
        family_weight: Global multiplier applied to this pack's total score.
        family_cap:   Maximum contribution (0.0–1.0) from this pack alone.
        description:  Human-readable purpose.
        source_refs:  Citation keys for provenance.
        version:      Pack-level version string.
    """
    name: str
    category: str
    target_layer: str  # e.g. 'prompt_signature', 'voice_dissonance', 'instruction_density'
    mode: str = 'both'
    patterns: Tuple[Tuple[str, float], ...] = ()
    keywords: FrozenSet[str] = frozenset()
    uppercase_keywords: FrozenSet[str] = frozenset()
    family_weight: float = 1.0
    family_cap: float = 1.0
    description: str = ''
    source_refs: Tuple[str, ...] = ()
    version: str = '1.0.0'


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 1: CONSTRAINT_FRAMES EXPANSION
# BCP 14 / RFC 2119 + EARS + Cardinality Operators
# ══════════════════════════════════════════════════════════════════════════════

PACK_OBLIGATION = LexiconPack(
    name='obligation',
    category='constraint',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='RFC 2119 / BCP 14 obligation operators. Uppercase forms carry '
                'normative meaning per RFC 8174.',
    source_refs=('RFC2119', 'RFC8174', 'BCP14'),
    uppercase_keywords=frozenset([
        'MUST', 'REQUIRED', 'SHALL',
    ]),
    patterns=(
        (r'\bmust\s+(?:always|ensure|verify|contain|produce|return|handle)\b', 1.5),
        (r'\b(?:is|are)\s+required\s+to\b', 1.2),
        (r'\bshall\s+(?:be|not|ensure|provide|include|comply)\b', 1.2),
        (r'\bmandatory\b', 1.0),
        (r'\bobligatory\b', 0.8),
        (r'\bnon-?negotiable\b', 1.0),
    ),
    keywords=frozenset([
        'must', 'required', 'shall', 'mandatory',
    ]),
    family_weight=1.2,
    family_cap=0.35,
)

PACK_PROHIBITION = LexiconPack(
    name='prohibition',
    category='constraint',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='RFC 2119 prohibition operators plus common task-prompt negation.',
    source_refs=('RFC2119', 'RFC8174'),
    uppercase_keywords=frozenset([
        'MUST NOT', 'SHALL NOT',
    ]),
    patterns=(
        (r'\bmust\s+not\b', 1.5),
        (r'\bshall\s+not\b', 1.2),
        (r'\bdo\s+not\b(?:\s+(?:include|use|add|modify|change|assume|omit|skip|invent))', 1.2),
        (r'\bnever\s+(?:use|include|add|assume|omit|modify|generate|invent|fabricate)\b', 1.3),
        (r'\b(?:is|are)\s+(?:not\s+)?(?:prohibited|forbidden|disallowed)\b', 1.0),
        (r'\bavoid\s+(?:using|including|adding|creating)\b', 0.8),
        (r'\bunder\s+no\s+circumstances?\b', 1.5),
        (r'\byou\s+may\s+not\b', 1.2),
        (r'\bmay\s+not\s+(?:be|introduce|omit|use|exceed|include|modify|alter)\b', 1.2),
    ),
    keywords=frozenset([
        'prohibited', 'forbidden', 'disallowed',
    ]),
    family_weight=1.2,
    family_cap=0.30,
)

PACK_RECOMMENDATION = LexiconPack(
    name='recommendation',
    category='constraint',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='RFC 2119 recommendation-tier operators (SHOULD/RECOMMENDED/MAY/OPTIONAL).',
    source_refs=('RFC2119', 'RFC8174'),
    uppercase_keywords=frozenset([
        'SHOULD', 'SHOULD NOT', 'RECOMMENDED', 'NOT RECOMMENDED',
        'MAY', 'OPTIONAL',
    ]),
    patterns=(
        (r'\bshould\s+(?:include|address|ensure|consider|provide|use|be)\b', 0.8),
        (r'\bshould\s+not\b', 0.9),
        (r'\b(?:it\s+is\s+)?recommended\s+(?:that|to)\b', 0.8),
        (r'\bnot\s+recommended\b', 0.8),
        (r'\boptional(?:ly)?\b', 0.6),
        (r'\b(?:may|can)\s+(?:optionally|also|additionally)\b', 0.5),
        (r'\bprefer(?:red|ably)\b', 0.6),
        (r'\bideally\b', 0.5),
    ),
    keywords=frozenset([
        'recommended', 'optional', 'preferably', 'ideally',
    ]),
    family_weight=0.8,
    family_cap=0.20,
)

PACK_CONDITIONAL = LexiconPack(
    name='conditional',
    category='constraint',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='EARS-derived event/state/conditional scaffolds for task specifications.',
    source_refs=('EARS', 'Mavin2009'),
    patterns=(
        # EARS: When [event], [system] shall...
        (r'\bwhen\s+(?:the|a|an)\s+\w+\s+(?:is|are|has|have|does|occurs?|triggers?|receives?|detects?)\b', 1.5),
        (r'\bwhen\s+\w+ing\b', 0.8),
        # EARS: While [state], [system] shall...
        (r'\bwhile\s+(?:the|a|an)\s+\w+\s+(?:is|are|remains?)\b', 1.3),
        (r'\bwhile\s+\w+ing\b', 0.8),
        # EARS: If [condition] then...
        (r'\bif\s+[^.]{5,60},?\s+then\b', 1.5),
        (r'\bif\s+(?:the|a|an|any|no)\s+\w+\s+(?:is|are|has|does|contains?|exceeds?|falls?|matches?)\b', 1.2),
        (r'\bif\s+present\b', 1.2),
        (r'\bif\s+absent\b', 1.2),
        (r'\bif\s+(?:not\s+)?(?:provided|specified|available|applicable|empty|null|blank|missing)\b', 1.3),
        # EARS: Where [feature] is included...
        (r'\bwhere\s+(?:the|a)\s+\w+\s+(?:is|are)\s+(?:included|enabled|supported|present|active)\b', 1.0),
        # General conditionals
        (r'\botherwise\b', 0.8),
        (r'\bunless\s+(?:the|a|an|otherwise|explicitly|specifically)\b', 1.0),
        (r'\bin\s+(?:the\s+)?(?:case|event)\s+(?:of|that|where)\b', 0.8),
        (r'\bprovided\s+that\b', 0.8),
        (r'\bgiven\s+that\b', 0.7),
    ),
    keywords=frozenset([
        'otherwise', 'unless', 'whereas', 'whenever',
    ]),
    family_weight=1.0,
    family_cap=0.30,
)

PACK_CARDINALITY = LexiconPack(
    name='cardinality',
    category='constraint',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='Cardinality and quantification operators for task constraints.',
    source_refs=('RFC2119', 'EARS'),
    patterns=(
        (r'\bexactly\s+\d+\b', 1.5),
        (r'\bat\s+least\s+\d+\b', 1.2),
        (r'\bat\s+most\s+\d+\b', 1.2),
        (r'\bno\s+more\s+than\s+\d+\b', 1.2),
        (r'\bno\s+fewer\s+than\s+\d+\b', 1.2),
        (r'\bno\s+less\s+than\b', 1.0),
        (r'\bbetween\s+\d+\s+and\s+\d+\b', 1.0),
        (r'\b(?:one|two|three|four|five)\s+(?:of\s+the\s+following|or\s+more|to\s+\w+)\b', 0.8),
        (r'\bone\s+of\b', 0.7),
        (r'\beach\s+(?:of\s+the|row|column|entry|item|record|field|section|task)\b', 1.0),
        (r'\bevery\s+(?:row|column|entry|item|record|field|section|task)\b', 1.0),
        (r'\ball\s+(?:of\s+the|rows|columns|entries|items|records|fields|sections|tasks)\b', 0.8),
        (r'\bonly\s+(?:one|the|if|when|those|items?|records?)\b', 0.8),
        (r'\bup\s+to\s+\d+\b', 0.8),
        (r'\bminimum\s+(?:of\s+)?\d+\b', 1.0),
        (r'\bmaximum\s+(?:of\s+)?\d+\b', 1.0),
        (r'\b(?:a\s+)?single\s+(?:row|column|entry|item|value|file|output)\b', 0.8),
        (r'\bper\s+(?:row|column|entry|item|record|patient|case|task)\b', 0.8),
    ),
    keywords=frozenset([
        'exactly', 'minimum', 'maximum',
    ]),
    family_weight=1.0,
    family_cap=0.25,
)

PACK_STATE = LexiconPack(
    name='state',
    category='constraint',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='State and lifecycle operators for task specifications.',
    source_refs=('EARS',),
    patterns=(
        (r'\b(?:initial|default|starting|baseline)\s+(?:state|value|condition|setting)\b', 1.0),
        (r'\b(?:final|end|terminal|completed?)\s+(?:state|value|condition|output)\b', 1.0),
        (r'\b(?:before|after|during|upon)\s+(?:processing|completion|submission|loading|saving|execution)\b', 0.8),
        (r'\b(?:on|upon)\s+(?:success|failure|error|timeout|completion|receipt)\b', 1.0),
        (r'\b(?:transition|change|switch|move)\s+(?:to|from|between)\s+(?:state|mode|phase)\b', 1.0),
        (r'\b(?:enabled?|disabled?|active|inactive|locked|unlocked|open|closed)\s+(?:state|mode|by default)\b', 0.8),
    ),
    keywords=frozenset([
        'initialized', 'finalized', 'pending', 'completed',
    ]),
    family_weight=0.8,
    family_cap=0.20,
)


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 2: SCHEMA / STRUCTURED-OUTPUT LEXICON
# JSON Schema + OpenAPI + Tabular Specs
# ══════════════════════════════════════════════════════════════════════════════

PACK_SCHEMA_JSON = LexiconPack(
    name='schema_json',
    category='schema',
    target_layer='voice_dissonance',
    mode='task_prompt',
    description='JSON Schema and data serialization vocabulary.',
    source_refs=('JSONSchema2020', 'OpenAPI3.1'),
    patterns=(
        (r'\bjson\s*schema\b', 2.0),
        (r'\b(?:json|yaml|xml|csv)\s+(?:output|format|file|object|response|payload)\b', 1.2),
        (r'\bschema\s+(?:for|of|with|object|definition|validation)\b', 1.5),
        (r'\b(?:request|response)\s+(?:body|schema|format|payload|object)\b', 1.2),
        (r'\b(?:query|path|header)\s+param(?:eter)?s?\b', 1.0),
        (r'\b(?:200|201|400|401|403|404|500)\s+(?:response|status|error)\b', 1.0),
        (r'\bendpoint\b', 0.8),
        (r'\bapi\s+(?:call|request|response|endpoint|contract|spec)\b', 1.2),
        (r'\bhttp\s+(?:get|post|put|patch|delete|method)\b', 1.0),
        (r'\brest(?:ful)?\s+api\b', 1.0),
    ),
    keywords=frozenset([
        'schema', 'endpoint', 'payload', 'serialization',
        'deserialization', 'marshalling', 'unmarshalling',
    ]),
    family_weight=1.0,
    family_cap=0.30,
)

PACK_SCHEMA_TYPES = LexiconPack(
    name='schema_types',
    category='schema',
    target_layer='voice_dissonance',
    mode='task_prompt',
    description='JSON Schema type system and property-control keywords.',
    source_refs=('JSONSchema2020',),
    patterns=(
        (r'\btype\s*:\s*["\']?(?:string|integer|number|boolean|array|object|null)\b', 1.5),
        (r'\b(?:required|optional)\s+(?:field|property|parameter|column|attribute)s?\b', 1.2),
        (r'\b(?:additional|pattern)\s*properties\b', 1.5),
        (r'\benum\s*:\s*\[', 1.5),
        (r'\b(?:enum|enumerat(?:ed?|ion))\s+(?:of|values?|type|field)\b', 1.2),
        (r'\b(?:min|max)(?:imum|Length|Items|Properties)\b', 1.0),
        (r'\bdefault\s*:\s', 1.0),
        (r'\bnullable\b', 1.0),
        (r'\b\$ref\b', 1.5),
        (r'\boneOf|anyOf|allOf\b', 1.5),
    ),
    keywords=frozenset([
        'nullable', 'properties', 'additionalproperties',
    ]),
    family_weight=1.0,
    family_cap=0.25,
)

PACK_DATA_FIELDS = LexiconPack(
    name='data_fields',
    category='schema',
    target_layer='voice_dissonance',
    mode='task_prompt',
    description='Generic data-structure vocabulary: field, key, value, record, etc.',
    source_refs=('JSONSchema2020', 'RFC4180', 'Frictionless'),
    patterns=(
        (r'\b(?:field|column|attribute)\s+(?:name|type|description|value|definition)s?\b', 1.0),
        (r'\bkey[- ]?value\s+pair\b', 1.2),
        (r'\bprimary\s+key\b', 1.0),
        (r'\bforeign\s+key\b', 1.0),
        (r'\bdata\s+(?:type|model|structure|format|contract|dictionary)\b', 1.0),
        (r'\b(?:input|output)\s+(?:field|column|schema|format|parameter)s?\b', 1.0),
        (r'\breturn\s+(?:type|value|format|schema)\b', 0.8),
    ),
    keywords=frozenset([
        'field', 'key', 'value', 'record', 'attribute',
    ]),
    family_weight=0.8,
    family_cap=0.20,
)

PACK_TABULAR = LexiconPack(
    name='tabular',
    category='schema',
    target_layer='voice_dissonance',
    mode='task_prompt',
    description='Tabular data and spreadsheet vocabulary (RFC 4180, Frictionless).',
    source_refs=('RFC4180', 'Frictionless'),
    patterns=(
        (r'\bcsv\s+(?:file|format|output|input|data|export|import)\b', 1.0),
        (r'\bdelimiter\b', 1.2),
        (r'\bheader\s+row\b', 1.2),
        (r'\b(?:first|top)\s+row\s+(?:is|contains?|should|as)\s+(?:the\s+)?header\b', 1.2),
        (r'\bworksheet\b', 1.0),
        (r'\btabular\s+(?:data|format|output|resource)\b', 1.2),
        (r'\bpivot\s+table\b', 1.0),
        (r'\bvlookup\b', 1.0),
        (r'\bspreadsheet\b', 0.8),
        (r'\btext/csv\b', 1.5),
        (r'\btsv\b', 0.8),
    ),
    keywords=frozenset([
        'delimiter', 'worksheet', 'spreadsheet', 'tabular',
    ]),
    family_weight=0.8,
    family_cap=0.20,
)


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 3: EXECUTABLE-SPEC / RUBRIC / GHERKIN
# ══════════════════════════════════════════════════════════════════════════════

PACK_GHERKIN = LexiconPack(
    name='gherkin',
    category='exec_spec',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='Gherkin BDD specification keywords. These structure executable '
                'specifications — very close to the prompt genre being detected.',
    source_refs=('Cucumber', 'Gherkin'),
    patterns=(
        (r'^\s*Feature:\s', 2.0),
        (r'^\s*Scenario(?:\s+Outline)?:\s', 2.0),
        (r'^\s*Given\s+', 1.5),
        (r'^\s*When\s+', 1.2),
        (r'^\s*Then\s+', 1.5),
        (r'^\s*And\s+', 0.5),
        (r'^\s*But\s+', 0.5),
        (r'^\s*Examples:\s', 1.5),
        (r'^\s*Background:\s', 1.5),
        (r'\b(?:given|when|then)\s+(?:the|a|an)\s+\w+\s+(?:is|are|has|does|should|shall)\b', 1.0),
    ),
    keywords=frozenset(),  # Gherkin keywords are positional — use patterns only
    family_weight=1.3,
    family_cap=0.30,
)

PACK_RUBRIC = LexiconPack(
    name='rubric',
    category='exec_spec',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='Rubric, evaluation, and grading vocabulary for assessment tasks.',
    source_refs=('GDPval', 'Prometheus'),
    patterns=(
        (r'\brubric\b', 1.5),
        (r'\bscoring\s+(?:criteria|rubric|guide|matrix|scale)\b', 1.5),
        (r'\bpass(?:ing)?[/\\]fail(?:ing)?\b', 1.5),
        (r'\bgrader?\b', 1.0),
        (r'\bchecklist\b', 1.0),
        (r'\bverification\s+(?:step|criteria|check|point)\b', 1.2),
        (r'\bvalidation\s+(?:step|criteria|check|rule)\b', 1.0),
        (r'\btest\s+case\b', 1.2),
        (r'\bedge\s+case\b', 1.2),
        (r'\bcorner\s+case\b', 1.0),
        (r'\bexpected\s+(?:output|result|response|behavior|value|answer)\b', 1.5),
        (r'\bgolden\s+(?:answer|response|output|standard|reference)\b', 1.5),
        (r'\bsource\s+of\s+truth\b', 1.5),
        (r'\bground\s+truth\b', 1.5),
        (r'\bgrounded\s+in\b', 1.0),
        (r'\bcite\s+(?:source|evidence|reference)\b', 1.0),
        (r'\bevidence[- ]based\b', 0.8),
        (r'\bscenario\b', 0.6),
        (r'\bexamples?\s*:\s', 0.8),
        (r'\bsample\s+(?:input|output|response|answer)\b', 1.2),
    ),
    keywords=frozenset([
        'rubric', 'checklist', 'grader', 'verification', 'validation',
    ]),
    family_weight=1.2,
    family_cap=0.30,
)

PACK_ACCEPTANCE = LexiconPack(
    name='acceptance',
    category='exec_spec',
    target_layer='prompt_signature',
    mode='task_prompt',
    description='Acceptance criteria and definition-of-done vocabulary.',
    source_refs=('Cucumber', 'AgileAlliance'),
    patterns=(
        (r'\bacceptance\s+criteria\b', 2.0),
        (r'\bdefinition\s+of\s+done\b', 1.5),
        (r'\bdone\s+(?:when|criteria|definition)\b', 1.0),
        (r'\bexit\s+criteria\b', 1.2),
        (r'\bentry\s+criteria\b', 1.0),
        (r'\bsuccess\s+(?:criteria|metric|condition|measure)\b', 1.2),
        (r'\bcompletion\s+criteria\b', 1.2),
        (r'\bdeliverable\s+(?:must|should|criteria|requirement)\b', 1.0),
        (r'\buser\s+story\b', 0.8),
        (r'\bas\s+a\s+\w+,?\s+i\s+want\b', 1.5),
    ),
    keywords=frozenset([
        'deliverable', 'milestone', 'criterion', 'criteria',
    ]),
    family_weight=1.0,
    family_cap=0.25,
)


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 4 (PARTIAL): IDI TYPED INSTRUCTION OPERATORS
# Task verbs, value-domain, formatting verbs
# ══════════════════════════════════════════════════════════════════════════════

PACK_TASK_VERBS = LexiconPack(
    name='task_verbs',
    category='instruction',
    target_layer='instruction_density',
    mode='task_prompt',
    description='Bloom-taxonomy action verbs for task specifications. Noisy alone; '
                'gains signal when paired with constraint/schema operators.',
    source_refs=('BloomTaxonomy', 'Utica'),
    patterns=(
        # Higher-order verbs (more signal)
        (r'\b(?:classify|categorize)\s+(?:the|each|all|every)\b', 1.2),
        (r'\b(?:evaluate|assess|critique|analyze)\s+(?:the|each|whether|how)\b', 1.2),
        (r'\b(?:design|architect|propose)\s+(?:a|an|the)\b', 1.0),
        (r'\b(?:justify|defend|argue|explain\s+why)\b', 1.0),
        # Mid-level verbs
        (r'\b(?:compare|contrast|differentiate)\s+(?:the|between|across)\b', 1.0),
        (r'\b(?:summarize|synthesize|consolidate)\s+(?:the|all|key)\b', 0.8),
        (r'\b(?:rewrite|revise|edit|refactor)\s+(?:the|this|each|to)\b', 0.8),
        (r'\b(?:translate|convert|transform)\s+(?:the|each|all|from|into|to)\b', 1.0),
        # Extraction / manipulation verbs
        (r'\b(?:extract|identify|locate|find|detect)\s+(?:the|all|each|any|every)\b', 1.0),
        (r'\b(?:label|tag|annotate|mark)\s+(?:the|each|all|every)\b', 1.0),
        (r'\b(?:rank|sort|order|prioritize)\s+(?:the|by|based|according)\b', 1.0),
        (r'\b(?:populate|fill|complete)\s+(?:the|each|all|every|a)\b', 0.8),
        (r'\b(?:validate|verify|check|confirm)\s+(?:that|the|each|whether|all)\b', 1.0),
        (r'\b(?:normalize|standardize|clean)\s+(?:the|all|each)\b', 1.0),
        (r'\b(?:parse|tokenize|split)\s+(?:the|each|into)\b', 1.0),
        (r'\b(?:map|associate|link|cross-reference)\s+(?:the|each|to|between)\b', 0.8),
        (r'\b(?:generate|produce|create|output)\s+(?:a|an|the|your)\b', 0.8),
        (r'\b(?:format|structure|organize)\s+(?:the|your|as|into|according)\b', 0.8),
    ),
    keywords=frozenset([
        'classify', 'identify', 'extract', 'label', 'compare', 'evaluate',
        'rewrite', 'translate', 'summarize', 'justify', 'rank', 'design',
        'generate', 'format', 'populate', 'validate', 'convert', 'normalize',
        'parse', 'map', 'annotate', 'synthesize', 'categorize', 'prioritize',
    ]),
    family_weight=0.7,  # Low weight alone — gains via pairing
    family_cap=0.20,
)

PACK_VALUE_DOMAIN = LexiconPack(
    name='value_domain',
    category='instruction',
    target_layer='instruction_density',
    mode='task_prompt',
    description='Value-domain and control operators (null handling, defaults, allowed values).',
    source_refs=('JSONSchema2020', 'RFC2119'),
    patterns=(
        (r'\b(?:true|false)\b', 0.5),
        (r'\b(?:yes|no)\b(?!\s+(?:longer|one|matter|idea))', 0.5),
        (r'\bnull\b', 1.0),
        (r'\bnone\b(?=\s|$|[,;.])', 0.5),
        (r'\bunknown\b', 0.5),
        (r'\bleave\s+blank\b', 1.5),
        (r'\bdefault\s+(?:to|value|is|of)\b', 1.2),
        (r'\bfallback\s+(?:to|value|is)\b', 1.2),
        (r'\ballowed\s+values?\b', 1.5),
        (r'\bvalid\s+values?\b', 1.5),
        (r'\bacceptable\s+values?\b', 1.2),
        (r'\bpermitted\s+values?\b', 1.2),
        (r'\bone\s+of\s*[:\[({]', 1.5),
        (r'\breturn\s+as\b', 1.2),
        (r'\boutput\s+as\b', 1.0),
        (r'\bformat\s+as\b', 1.0),
        (r'\bMISSING\b', 2.0),
        (r'\bN/?A\b', 0.5),
        (r'\bempty\s+(?:string|value|cell|field)\b', 1.0),
        (r'\bplaceholder\b', 0.8),
        (r'\bsentinel\s+value\b', 1.5),
    ),
    keywords=frozenset([
        'null', 'fallback', 'default', 'placeholder', 'sentinel',
    ]),
    family_weight=1.0,
    family_cap=0.25,
)


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 5 (PARTIAL): FORMAT / MARKUP SUBLEXICON
# ══════════════════════════════════════════════════════════════════════════════

PACK_FORMAT_MARKUP = LexiconPack(
    name='format_markup',
    category='format',
    target_layer='voice_dissonance',
    mode='task_prompt',
    description='Markdown, code fences, tables, task lists — output-shape markers '
                'that occur constantly in modern task prompts.',
    source_refs=('CommonMark', 'GFM'),
    patterns=(
        (r'```\w*\n', 1.5),   # Fenced code block
        (r'~~~\w*\n', 1.2),   # Tilde code fence
        (r'\bcode\s+(?:block|fence|snippet|example)\b', 1.0),
        (r'\bmarkdown\s+(?:format|table|output|syntax)\b', 1.2),
        (r'\b(?:bullet|numbered|ordered|unordered)\s+list\b', 0.8),
        (r'\btask\s+list\b', 1.0),
        (r'\bchecklist\s+(?:format|style|item)\b', 1.0),
        (r'\[[ x]\]', 1.5),   # GitHub task list syntax
        (r'\bheading\s+(?:level|format|style)\b', 0.8),
        (r'\btable\s+(?:format|with|header|row|column)\b', 1.0),
        (r'\bheader\s+row\b', 1.0),
        (r'\bpipe[- ]?delimited\b', 1.2),
        (r'\binline\s+code\b', 0.8),
        (r'\bformatted?\s+(?:as|in|using)\s+(?:markdown|json|yaml|csv|xml|html)\b', 1.2),
        (r'\boutput\s+format\s*:', 1.5),
        (r'\bresponse\s+format\s*:', 1.5),
    ),
    keywords=frozenset([
        'markdown', 'backtick', 'codeblock',
    ]),
    family_weight=0.9,
    family_cap=0.20,
)


# ══════════════════════════════════════════════════════════════════════════════
# PACK REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

PACK_REGISTRY: Dict[str, LexiconPack] = {
    # Priority 1: Constraint families
    'obligation': PACK_OBLIGATION,
    'prohibition': PACK_PROHIBITION,
    'recommendation': PACK_RECOMMENDATION,
    'conditional': PACK_CONDITIONAL,
    'cardinality': PACK_CARDINALITY,
    'state': PACK_STATE,
    # Priority 2: Schema / structured output
    'schema_json': PACK_SCHEMA_JSON,
    'schema_types': PACK_SCHEMA_TYPES,
    'data_fields': PACK_DATA_FIELDS,
    'tabular': PACK_TABULAR,
    # Priority 3: Executable spec / rubric
    'gherkin': PACK_GHERKIN,
    'rubric': PACK_RUBRIC,
    'acceptance': PACK_ACCEPTANCE,
    # Priority 4: IDI typed operators
    'task_verbs': PACK_TASK_VERBS,
    'value_domain': PACK_VALUE_DOMAIN,
    # Priority 5: Format / markup
    'format_markup': PACK_FORMAT_MARKUP,
}


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _compile_pack(pack: LexiconPack):
    """Compile regex patterns for a pack. Returns list of (compiled_re, weight)."""
    compiled = []
    flags = re.IGNORECASE | re.MULTILINE
    for pat_str, weight in pack.patterns:
        try:
            compiled.append((re.compile(pat_str, flags), weight))
        except re.error as e:
            print(f"WARNING: Bad regex in pack '{pack.name}': {pat_str} — {e}")
    return compiled


# Pre-compile all packs at import time
_COMPILED_PACKS: Dict[str, list] = {}
_KEYWORD_RES: Dict[str, re.Pattern] = {}
_UPPERCASE_RES: Dict[str, re.Pattern] = {}

for _name, _pack in PACK_REGISTRY.items():
    _COMPILED_PACKS[_name] = _compile_pack(_pack)

    if _pack.keywords:
        _kw_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in sorted(_pack.keywords)) + r')\b'
        _KEYWORD_RES[_name] = re.compile(_kw_pattern, re.IGNORECASE)

    if _pack.uppercase_keywords:
        # Case-sensitive matching for normative uppercase forms
        _uc_parts = []
        for kw in sorted(_pack.uppercase_keywords, key=len, reverse=True):
            _uc_parts.append(r'\b' + re.escape(kw) + r'\b')
        _UPPERCASE_RES[_name] = re.compile('|'.join(_uc_parts))


@dataclass
class PackScore:
    """Scoring result for a single pack against a text."""
    pack_name: str
    category: str
    raw_hits: int = 0
    weighted_hits: float = 0.0
    keyword_hits: int = 0
    uppercase_hits: int = 0
    raw_score: float = 0.0
    capped_score: float = 0.0
    matches: List[str] = field(default_factory=list)
    spans: List[dict] = field(default_factory=list)


def score_pack(text: str, pack_name: str, n_sentences: int = 1) -> PackScore:
    """Score a single pack against text.

    Args:
        text: Normalized text to score.
        pack_name: Key in PACK_REGISTRY.
        n_sentences: Sentence count for density normalization.

    Returns:
        PackScore with hits, weighted score (density-normalized), capped score,
        and character-level spans for each match.
    """
    pack = PACK_REGISTRY[pack_name]
    compiled = _COMPILED_PACKS[pack_name]
    n_sents = max(n_sentences, 1)

    result = PackScore(pack_name=pack_name, category=pack.category)

    # Pattern matching (finditer for span capture)
    for compiled_re, weight in compiled:
        for m in compiled_re.finditer(text):
            result.raw_hits += 1
            result.weighted_hits += weight
            if len(result.matches) < 3:
                result.matches.append(m.group())
            result.spans.append({
                'start': m.start(),
                'end': m.end(),
                'text': m.group()[:80],
                'pack': pack_name,
                'weight': weight,
            })

    # Keyword matching (finditer for span capture, no weight — counted separately)
    kw_re = _KEYWORD_RES.get(pack_name)
    if kw_re:
        for m in kw_re.finditer(text):
            result.keyword_hits += 1
            result.spans.append({
                'start': m.start(),
                'end': m.end(),
                'text': m.group(),
                'pack': pack_name,
                'weight': 0.0,
                'type': 'keyword',
            })

    # Uppercase keyword matching (case-sensitive, finditer for span capture)
    uc_re = _UPPERCASE_RES.get(pack_name)
    if uc_re:
        for m in uc_re.finditer(text):
            result.uppercase_hits += 1
            result.weighted_hits += 2.0
            result.spans.append({
                'start': m.start(),
                'end': m.end(),
                'text': m.group(),
                'pack': pack_name,
                'weight': 2.0,
                'type': 'uppercase',
            })

    # Density-normalized score: weighted_hits per sentence × family_weight
    density = result.weighted_hits / n_sents
    result.raw_score = density * pack.family_weight
    result.capped_score = min(result.raw_score, pack.family_cap)

    return result


def score_packs(text: str, pack_names: Optional[List[str]] = None,
                n_sentences: int = 1) -> Dict[str, PackScore]:
    """Score multiple packs against text.

    Args:
        text: Normalized text.
        pack_names: List of pack names to score. If None, scores all packs.
        n_sentences: Sentence count for density normalization.

    Returns:
        Dict mapping pack_name → PackScore.
    """
    names = pack_names or list(PACK_REGISTRY.keys())
    return {name: score_pack(text, name, n_sentences) for name in names}


# ══════════════════════════════════════════════════════════════════════════════
# SPAN-LEVEL EXPLAINABILITY ("X-RAY" VIEW)
# ══════════════════════════════════════════════════════════════════════════════

def score_pack_spans(text: str, pack_name: str) -> List[Tuple[int, int, str, str, float]]:
    """Return character-level spans for all regex/keyword hits in a pack.

    Delegates to score_pack() and extracts span data from PackScore.spans.

    Returns:
        List of (start_char, end_char, matched_text, pack_name, weight) tuples,
        sorted by start_char.
    """
    ps = score_pack(text, pack_name)
    spans = [(s['start'], s['end'], s['text'], s['pack'], s['weight'])
             for s in ps.spans]
    spans.sort(key=lambda s: s[0])
    return spans


def score_all_pack_spans(text: str, pack_names: Optional[List[str]] = None
                         ) -> List[Tuple[int, int, str, str, float]]:
    """Return merged character-level spans across multiple packs.

    Args:
        text: The original (normalized) text.
        pack_names: Pack names to scan. If None, scans all registered packs.

    Returns:
        Sorted list of (start_char, end_char, matched_text, pack_name, weight).
    """
    names = pack_names or list(PACK_REGISTRY.keys())
    all_spans = []
    for name in names:
        all_spans.extend(score_pack_spans(text, name))
    all_spans.sort(key=lambda s: s[0])
    return all_spans


def get_packs_for_layer(target_layer: str) -> List[str]:
    """Get pack names that feed a specific pipeline layer."""
    return [name for name, pack in PACK_REGISTRY.items()
            if pack.target_layer == target_layer or pack.target_layer == 'multi']


def get_packs_for_mode(mode: str) -> List[str]:
    """Get pack names eligible for a detection mode."""
    return [name for name, pack in PACK_REGISTRY.items()
            if pack.mode == mode or pack.mode == 'both']


def get_category_score(pack_scores: Dict[str, PackScore], category: str) -> float:
    """Aggregate capped scores across all packs in a category."""
    return sum(
        ps.capped_score for ps in pack_scores.values()
        if ps.category == category
    )


def get_total_constraint_score(pack_scores: Dict[str, PackScore]) -> float:
    """Sum of all constraint-family capped scores (Priority 1)."""
    return get_category_score(pack_scores, 'constraint')


def get_total_schema_score(pack_scores: Dict[str, PackScore]) -> float:
    """Sum of all schema-family capped scores (Priority 2)."""
    return get_category_score(pack_scores, 'schema')


def get_total_exec_spec_score(pack_scores: Dict[str, PackScore]) -> float:
    """Sum of all exec_spec-family capped scores (Priority 3)."""
    return get_category_score(pack_scores, 'exec_spec')


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_pack_enhanced_cfd(text: str, n_sentences: int,
                               legacy_cfd: float = 0.0) -> dict:
    """Compute pack-enhanced Constraint Frame Density for prompt_signature.

    Combines legacy CONSTRAINT_FRAMES patterns with the new typed packs.
    Returns dict with enhanced_cfd, pack breakdown, and legacy comparison.
    """
    constraint_packs = get_packs_for_layer('prompt_signature')
    constraint_packs = [p for p in constraint_packs
                        if PACK_REGISTRY[p].category in ('constraint', 'exec_spec')]

    scores = score_packs(text, constraint_packs, n_sentences)

    total_constraint = get_total_constraint_score(scores)
    total_exec_spec = get_total_exec_spec_score(scores)

    # Enhanced CFD: legacy + new packs (with diminishing returns)
    pack_boost = total_constraint * 0.6 + total_exec_spec * 0.4
    enhanced_cfd = legacy_cfd + pack_boost

    return {
        'enhanced_cfd': round(enhanced_cfd, 4),
        'legacy_cfd': legacy_cfd,
        'pack_constraint_score': round(total_constraint, 4),
        'pack_exec_spec_score': round(total_exec_spec, 4),
        'pack_boost': round(pack_boost, 4),
        'active_packs': {name: {
            'hits': s.raw_hits,
            'weighted': round(s.weighted_hits, 2),
            'capped': round(s.capped_score, 4),
            'uc_hits': s.uppercase_hits,
        } for name, s in scores.items() if s.raw_hits > 0},
        'distinct_pack_families': sum(1 for s in scores.values() if s.raw_hits > 0),
    }


def compute_pack_enhanced_spec(text: str, n_sentences: int,
                                legacy_spec_score: float = 0.0) -> dict:
    """Compute pack-enhanced spec_score for voice_dissonance.

    Adds schema/structured-output vocabulary to the legacy spreadsheet-focused spec_score.
    """
    schema_packs = get_packs_for_layer('voice_dissonance')
    schema_packs = [p for p in schema_packs
                    if PACK_REGISTRY[p].category in ('schema', 'format')]

    scores = score_packs(text, schema_packs, n_sentences)

    total_schema = get_total_schema_score(scores)
    format_score = get_category_score(scores, 'format')

    pack_boost = total_schema * 0.7 + format_score * 0.3
    enhanced_spec = legacy_spec_score + pack_boost

    return {
        'enhanced_spec': round(enhanced_spec, 4),
        'legacy_spec': legacy_spec_score,
        'pack_schema_score': round(total_schema, 4),
        'pack_format_score': round(format_score, 4),
        'pack_boost': round(pack_boost, 4),
        'active_packs': {name: {
            'hits': s.raw_hits,
            'weighted': round(s.weighted_hits, 2),
            'capped': round(s.capped_score, 4),
        } for name, s in scores.items() if s.raw_hits > 0},
    }


def compute_pack_enhanced_idi(text: str, n_words: int,
                               legacy_idi: float = 0.0) -> dict:
    """Compute pack-enhanced IDI for instruction_density.

    Adds typed task-verb and value-domain operators to legacy imperative counting.
    Key insight from roadmap: action verbs alone are noisy, but action verbs
    PLUS constraint/schema operators are strong signal.
    """
    idi_packs = get_packs_for_layer('instruction_density')
    per100 = max(n_words / 100, 1)

    scores = score_packs(text, idi_packs, n_sentences=1)

    # Raw per-100-word density for IDI compatibility
    task_verb_density = scores.get('task_verbs', PackScore('task_verbs', 'instruction')).weighted_hits / per100
    value_domain_density = scores.get('value_domain', PackScore('value_domain', 'instruction')).weighted_hits / per100

    # Pairing bonus: verbs get extra weight when constraint/schema packs also fired
    # (Computed by caller who has access to L2.5 and L2.6 results)
    pack_idi_contribution = (task_verb_density * 1.0 + value_domain_density * 2.0)
    enhanced_idi = legacy_idi + pack_idi_contribution

    return {
        'enhanced_idi': round(enhanced_idi, 2),
        'legacy_idi': legacy_idi,
        'task_verb_density': round(task_verb_density, 3),
        'value_domain_density': round(value_domain_density, 3),
        'pack_contribution': round(pack_idi_contribution, 2),
        'active_packs': {name: {
            'hits': s.raw_hits,
            'weighted': round(s.weighted_hits, 2),
            'capped': round(s.capped_score, 4),
        } for name, s in scores.items() if s.raw_hits > 0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def pack_summary() -> str:
    """Print a summary table of all registered packs."""
    lines = [
        f"{'Pack':<20} {'Category':<12} {'Layer':<6} {'Mode':<14} "
        f"{'Patterns':<9} {'Keywords':<9} {'UC Keys':<8} {'Weight':<7} {'Cap':<5}",
        '-' * 100,
    ]
    for name, pack in PACK_REGISTRY.items():
        lines.append(
            f"{name:<20} {pack.category:<12} {pack.target_layer:<6} {pack.mode:<14} "
            f"{len(pack.patterns):<9} {len(pack.keywords):<9} {len(pack.uppercase_keywords):<8} "
            f"{pack.family_weight:<7.1f} {pack.family_cap:<5.2f}"
        )
    return '\n'.join(lines)


def diagnose_text(text: str, n_sentences: int = 1) -> str:
    """Score all packs and return a diagnostic report."""
    scores = score_packs(text, n_sentences=n_sentences)

    lines = [f"BEET Lexicon Pack Diagnostic (v{__version__})", '=' * 70]

    for cat in ['constraint', 'schema', 'exec_spec', 'instruction', 'format']:
        cat_packs = [(n, s) for n, s in scores.items() if s.category == cat]
        if not cat_packs:
            continue

        cat_total = sum(s.capped_score for _, s in cat_packs)
        lines.append(f"\n{cat.upper()} (total capped: {cat_total:.3f})")
        lines.append('-' * 50)

        for name, s in sorted(cat_packs, key=lambda x: -x[1].capped_score):
            if s.raw_hits > 0:
                lines.append(
                    f"  {name:<20} hits={s.raw_hits:<3} kw={s.keyword_hits:<3} "
                    f"uc={s.uppercase_hits:<2} raw={s.raw_score:.3f} "
                    f"cap={s.capped_score:.3f}  {s.matches[:2]}"
                )

    return '\n'.join(lines)


if __name__ == '__main__':
    print(pack_summary())
    print()

    # Quick self-test with a synthetic LLM-style task prompt
    test_text = """
    You are a senior data engineer. You MUST process the input CSV file containing
    patient records. Each row MUST have a valid patient_id field. If the field is
    absent or null, leave blank and flag as MISSING. The output schema MUST include:
    patient_id (string, required), diagnosis_code (enum: ["A01", "B02", "C03"]),
    and risk_score (number, between 0 and 100).

    Given the input file has header row, When processing each record, Then validate
    all required fields. If any field fails validation, mark the row with
    validation_status = "FAIL". Expected output: a CSV with exactly 5 columns.

    Acceptance criteria:
    1. All rows must pass schema validation
    2. No more than 5% of rows may have MISSING values
    3. Output must be formatted as pipe-delimited CSV

    Rubric: Pass/fail based on completeness and accuracy of the output.
    Edge cases: empty input file, duplicate patient_ids, malformed dates.
    """

    sentences = [s.strip() for s in test_text.strip().split('.') if len(s.strip()) > 10]
    print(diagnose_text(test_text, n_sentences=len(sentences)))
