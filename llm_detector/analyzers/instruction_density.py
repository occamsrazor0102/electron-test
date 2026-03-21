"""Instruction Density Index (IDI) -- catches formal-exhaustive LLM output."""

import re


def run_instruction_density(text):
    """Compute instruction density index. Returns dict of metrics."""
    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    imp_keywords = ['must', 'include', 'create', 'load', 'set', 'show', 'use', 'derive', 'treat', 'mark']
    imperatives = sum(len(re.findall(r'\b' + kw + r'\b', text, re.IGNORECASE)) for kw in imp_keywords)

    cond_keywords = ['if', 'otherwise', 'when', 'unless']
    conditionals = sum(len(re.findall(r'\b' + kw + r'\b', text, re.IGNORECASE)) for kw in cond_keywords)

    binary_specs = len(re.findall(r'\b(?:Yes|No)\b', text))
    missing_handling = len(re.findall(r'\bMISSING\b', text))
    flag_count = len(re.findall(r'\b[Ff]lag\b', text))

    idi = (imperatives * 1.0 + conditionals * 2.0 + binary_specs * 1.5 +
           missing_handling * 3.0 + flag_count * 2.0) / per100

    return {
        'idi': idi,
        'imperatives': imperatives,
        'imp_rate': imperatives / per100,
        'conditionals': conditionals,
        'cond_rate': conditionals / per100,
        'binary_specs': binary_specs,
        'missing_refs': missing_handling,
        'flag_count': flag_count,
    }
