#!/usr/bin/env python3
"""Build a single-file distribution from the llm_detector package.

Reads all package modules in dependency order, strips internal imports,
deduplicates external imports, and writes dist_single/llm_detector.py.

Usage:
    python _build_monolith.py
"""

import os
import re
import textwrap

REPO = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(REPO, 'llm_detector')
OUT_DIR = os.path.join(REPO, 'dist_single')
OUT_FILE = os.path.join(OUT_DIR, 'llm_detector.py')

# Topological order: dependencies before dependents.
# Each entry is (section_label, relative_path_from_PKG).
# __init__.py and __main__.py are skipped — their re-exports are unnecessary.
MODULE_ORDER = [
    # Foundation
    ('compat',                  'compat.py'),
    ('text_utils',              'text_utils.py'),
    ('language_gate',           'language_gate.py'),
    ('normalize',               'normalize.py'),
    ('io',                      'io.py'),
    ('similarity',              'similarity.py'),
    ('calibration',             'calibration.py'),
    ('baselines',               'baselines.py'),
    # Analyzers (order: leaf → dependents)
    ('analyzers/preamble',      'analyzers/preamble.py'),
    ('analyzers/fingerprint',   'analyzers/fingerprint.py'),
    ('analyzers/prompt_signature', 'analyzers/prompt_signature.py'),
    ('analyzers/voice_dissonance', 'analyzers/voice_dissonance.py'),
    ('analyzers/instruction_density', 'analyzers/instruction_density.py'),
    ('analyzers/semantic_resonance', 'analyzers/semantic_resonance.py'),
    ('analyzers/self_similarity', 'analyzers/self_similarity.py'),
    ('analyzers/continuation_api', 'analyzers/continuation_api.py'),
    ('analyzers/continuation_local', 'analyzers/continuation_local.py'),
    ('analyzers/perplexity',    'analyzers/perplexity.py'),
    ('analyzers/stylometry',    'analyzers/stylometry.py'),
    ('analyzers/windowing',     'analyzers/windowing.py'),
    # Lexicon
    ('lexicon/packs',           'lexicon/packs.py'),
    ('lexicon/integration',     'lexicon/integration.py'),
    # Channels
    ('channels/__init__',       'channels/__init__.py'),
    ('channels/prompt_structure', 'channels/prompt_structure.py'),
    ('channels/stylometric',    'channels/stylometric.py'),
    ('channels/continuation',   'channels/continuation.py'),
    ('channels/windowed',       'channels/windowed.py'),
    # Fusion & pipeline
    ('fusion',                  'fusion.py'),
    ('pipeline',                'pipeline.py'),
    # UI
    ('gui',                     'gui.py'),
    ('cli',                     'cli.py'),
]

# Regex to match internal imports
_INTERNAL_IMPORT_RE = re.compile(
    r'^\s*(from\s+llm_detector[\w.]*\s+import\s+.*|import\s+llm_detector[\w.]*).*$'
)

# Regex to match any import line (for deduplication)
_IMPORT_RE = re.compile(
    r'^(import\s+\S+|from\s+\S+\s+import\s+.*)$'
)

# Regex to match conditional internal imports like "if HAS_SPACY:\n    from llm_detector..."
_COND_INTERNAL_RE = re.compile(
    r'^(\s*if\s+\w+:\s*)$'
)


def strip_internal_imports(lines):
    """Remove all `from llm_detector...` and `import llm_detector...` lines.
    Handles multi-line imports (with parentheses) and conditional blocks.
    """
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is a conditional guard followed by an internal import
        if _COND_INTERNAL_RE.match(line) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if _INTERNAL_IMPORT_RE.match(next_line):
                i += 2
                # Skip continuation of multi-line import
                if '(' in next_line and ')' not in next_line:
                    while i < len(lines) and ')' not in lines[i - 1]:
                        i += 1
                while i < len(lines) and _INTERNAL_IMPORT_RE.match(lines[i]):
                    i += 1
                continue
        if _INTERNAL_IMPORT_RE.match(line):
            # Multi-line import: `from llm_detector.x import (`
            if '(' in line and ')' not in line:
                i += 1
                while i < len(lines) and ')' not in lines[i]:
                    i += 1
                i += 1  # skip the closing )
            else:
                i += 1
            continue
        result.append(line)
        i += 1
    return result


def extract_external_imports(lines):
    """Extract external import lines and return (imports, body_lines)."""
    imports = []
    body = []
    in_header = True
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Stay in header while we see imports, comments, blanks, docstrings
        if in_header:
            if stripped == '' or stripped.startswith('#'):
                i += 1
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Skip docstring
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    i += 1
                    continue
                else:
                    i += 1
                    while i < len(lines) and quote not in lines[i]:
                        i += 1
                    i += 1  # skip closing line
                    continue
            if _IMPORT_RE.match(stripped):
                imports.append(stripped)
                i += 1
                continue
            # Conditional import block at top level
            if re.match(r'^if\s+(HAS_\w+|__name__)', stripped):
                block = [line]
                i += 1
                while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip() == ''):
                    if lines[i].strip():
                        block.append(lines[i])
                    i += 1
                # Check if block contains imports
                block_imports = [l.strip() for l in block[1:] if _IMPORT_RE.match(l.strip())]
                if block_imports:
                    # Keep as conditional import block in body (e.g., `if HAS_SEMANTIC: from sklearn...`)
                    for bl in block:
                        body.append(bl)
                    body.append('')
                continue
            in_header = False
        body.append(line)
        i += 1
    return imports, body


def read_module(path):
    """Read a module file and return its lines."""
    with open(path, 'r') as f:
        return f.read().splitlines()


def fixup_special_cases(section, lines):
    """Handle known naming collisions and special patterns."""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # lexicon/packs.py has its own __version__ and __pack_date__
        if section == 'lexicon/packs':
            line = line.replace("__version__ = ", "_PACKS_VERSION = ")
            line = line.replace("__pack_date__ = ", "_PACKS_DATE = ")
            line = re.sub(r'\b__version__\b(?!\s*=)', '_PACKS_VERSION', line)
            line = re.sub(r'\b__pack_date__\b(?!\s*=)', '_PACKS_DATE', line)
        # lexicon/integration.py uses lp. prefix via `import llm_detector.lexicon.packs as lp`
        if section == 'lexicon/integration':
            line = re.sub(r'\blp\.', '', line)
            # Replace try/except ImportError block with just _HAS_ANALYZERS = True
            if line.strip() == 'try:':
                # Look ahead for except ImportError
                block_end = i + 1
                while block_end < len(lines):
                    if lines[block_end].strip().startswith('except'):
                        break
                    block_end += 1
                if block_end < len(lines) and 'ImportError' in lines[block_end]:
                    # Skip try block, emit _HAS_ANALYZERS = True, skip except block
                    result.append('_HAS_ANALYZERS = True')
                    # Skip past except block
                    block_end += 1
                    while block_end < len(lines) and (lines[block_end].startswith('    ') or not lines[block_end].strip()):
                        block_end += 1
                    i = block_end
                    continue
        # channels/__init__.py defines ChannelResult — keep the class, strip re-exports
        if section == 'channels/__init__':
            if line.strip().startswith('from llm_detector.channels.'):
                i += 1
                continue
        result.append(line)
        i += 1
    return result


def remove_if_name_main(lines):
    """Remove `if __name__ == '__main__':` blocks except from cli section."""
    result = []
    skip_block = False
    for line in lines:
        if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]", line):
            skip_block = True
            continue
        if skip_block:
            if line and not line[0].isspace() and line.strip():
                skip_block = False
                result.append(line)
            continue
        result.append(line)
    return result


def build():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_external_imports = set()
    conditional_imports = []
    sections = []

    for section_label, rel_path in MODULE_ORDER:
        path = os.path.join(PKG, rel_path)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {rel_path}")
            continue

        lines = read_module(path)
        lines = strip_internal_imports(lines)
        lines = fixup_special_cases(section_label, lines)

        # Don't strip if __name__ from cli — we need it for entry point
        if section_label != 'cli':
            lines = remove_if_name_main(lines)

        ext_imports, body = extract_external_imports(lines)
        all_external_imports.update(ext_imports)

        # Clean up leading/trailing blank lines in body
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()

        if body:
            sections.append((section_label, body))
            print(f"  OK: {rel_path} ({len(body)} lines)")

    # ── Assemble output ──
    out_lines = []

    # Header
    out_lines.append('#!/usr/bin/env python3')
    out_lines.append('"""')
    out_lines.append('LLM-Generated Task Prompt Detection Pipeline — Single-File Distribution')
    out_lines.append('=' * 70)
    out_lines.append('Auto-generated by _build_monolith.py from the llm_detector package.')
    out_lines.append('Contains all detection layers, channels, fusion, pipeline, CLI, and GUI.')
    out_lines.append('')
    out_lines.append('Features (monolith-only):')
    out_lines.append('  - Chain-of-thought leakage detection (<think> tags, reasoning-model phrases)')
    out_lines.append('  - DivEye-inspired surprisal variance and volatility decay in perplexity analysis')
    out_lines.append('  - Simpler file structure optimized for single-file distribution')
    out_lines.append('"""')
    out_lines.append('')
    out_lines.append("__version__ = '0.61.0'")
    out_lines.append('')

    # Deduplicated imports — sort stdlib first, then third-party
    stdlib = set()
    thirdparty = set()
    STDLIB_MODS = {
        're', 'os', 'sys', 'json', 'math', 'logging', 'argparse', 'unicodedata',
        'collections', 'hashlib', 'threading', 'pathlib', 'textwrap', 'string',
        'functools', 'itertools', 'copy', 'datetime', 'io', 'abc', 'typing',
    }
    for imp in sorted(all_external_imports):
        # Extract module name
        m = re.match(r'(?:from\s+(\S+)|import\s+(\S+))', imp)
        if m:
            mod = (m.group(1) or m.group(2)).split('.')[0]
            if mod in STDLIB_MODS:
                stdlib.add(imp)
            else:
                thirdparty.add(imp)

    out_lines.append('# ── Standard library ─────────────────────────────────────────────────')
    for imp in sorted(stdlib):
        out_lines.append(imp)
    out_lines.append('')
    if thirdparty:
        out_lines.append('# ── Third-party (conditional) ────────────────────────────────────────')
        for imp in sorted(thirdparty):
            out_lines.append(imp)
        out_lines.append('')

    # Sections
    for section_label, body in sections:
        out_lines.append('')
        out_lines.append(f'# {"═" * 68}')
        out_lines.append(f'# ══ {section_label} {"═" * (63 - len(section_label))}')
        out_lines.append(f'# {"═" * 68}')
        out_lines.append('')
        out_lines.extend(body)
        out_lines.append('')

    # Entry point
    out_lines.append('')
    out_lines.append("if __name__ == '__main__':")
    out_lines.append('    main()')
    out_lines.append('')

    with open(OUT_FILE, 'w') as f:
        f.write('\n'.join(out_lines))

    total = len(out_lines)
    print(f"\n  Built {OUT_FILE} ({total} lines)")
    print(f"  Sections: {len(sections)}")


if __name__ == '__main__':
    print("Building single-file distribution...")
    build()
    print("Done.")
