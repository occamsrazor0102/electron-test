"""Voice-Specification Dissonance (VSD) + Sterile Spec Index detection."""

import re

CASUAL_MARKERS = [
    'hey', 'ok so', 'ok,', 'dont', 'wont', 'cant', 'gonna', 'gotta',
    'thx', 'pls', 'gimme', 'lemme', 'kinda', 'sorta', 'tho', 'btw',
    'fyi', 'alot', 'ya', 'yep', 'nah', 'nope', 'lol', 'haha',
]

MANUFACTURED_TYPOS = [
    'atached', 'alot', 'recieved', 'seperate', 'occured', 'wierd',
    'definately', 'accomodate', 'occurence', 'independant', 'noticable',
    'occassion', 'tommorow', 'calender', 'begining', 'acheive', 'untill',
    'beleive', 'existance', 'grammer', 'arguement', 'commited',
    'maintainance', 'necesary', 'occuring', 'persue', 'prefered',
    'recomend', 'refered', 'succesful', 'suprise',
]


def _build_marker_pattern(marker):
    tokens = marker.split()
    if len(tokens) > 1:
        escaped = r'\s+'.join(re.escape(t) for t in tokens)
    else:
        escaped = re.escape(marker)
    first_char = tokens[0][0]
    if first_char.isalnum() or first_char == '_':
        leading = r'\b'
    else:
        leading = r'(?<!\w)'
    last_char = tokens[-1][-1]
    if last_char.isalnum() or last_char == '_':
        trailing = r'\b'
    else:
        trailing = r'(?!\w)'
    return leading + escaped + trailing


_CASUAL_RE = [re.compile(_build_marker_pattern(m), re.IGNORECASE) for m in CASUAL_MARKERS]
_TYPO_RE = [re.compile(r'\b' + re.escape(t) + r'\b', re.IGNORECASE) for t in MANUFACTURED_TYPOS]
# Combined em-dash pattern (covers unicode dashes and spaced hyphens in one pass)
_EM_DASH_RE = re.compile(r'(?<!\d)\s?[—–]\s?(?!\d)| - ')


def run_voice_dissonance(text):
    """Detect voice-specification dissonance. Returns dict of metrics."""
    words = text.split()
    n_words = len(words)
    per100 = max(n_words / 100, 1)

    casual_count = sum(len(pat.findall(text)) for pat in _CASUAL_RE)
    misspelling_count = sum(len(pat.findall(text)) for pat in _TYPO_RE)

    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|s|d|ll|m)\b", text, re.IGNORECASE))

    em_dashes = len(_EM_DASH_RE.findall(text))

    lowercase_starts = sum(1 for line in text.split('\n') if line.strip() and line.strip()[0].islower())

    voice_score = (casual_count * 5 + misspelling_count * 1 + contractions * 1.5
                   + em_dashes * 1 + lowercase_starts * 0.5) / per100

    camel_cols = len(re.findall(r'[A-Z][a-z]+_[A-Z][a-z_]+', text))
    filenames = len(set(re.findall(
        r'\w+\.(?:csv|xlsx|xls|tsv|json|xml|pdf|docx|doc|pptx|ppt|txt|md|html|py|zip|png|jpg|jpeg|gif|mp4)\b',
        text, re.IGNORECASE)))
    calcs = len(re.findall(
        r'(calculated?|computed?|deriv|formula|multiply|divid|subtract|sum\b|average|ratio|percent|\bnet\b.*[-=])',
        text, re.IGNORECASE))
    tabs = len(re.findall(r'(?i)(tab \d|\btab\b.*[:—-]|sheet \d)', text))
    col_listings = len(re.findall(r'(?:columns?|fields?)\s*[:]\s*\w', text, re.IGNORECASE))
    tech_parens = len(re.findall(
        r'\([^)]*(?:\.\w{2,4}|%|\d+[kKmM]?\b|formula|column|tab)[^)]*\)', text))

    spec_score = (camel_cols * 1.5 + filenames * 2 + calcs * 2 + tabs * 3
                  + col_listings * 3 + tech_parens * 1) / per100

    vsd = voice_score * spec_score

    hedges = len(re.findall(
        r'\b(pretty sure|i think|probably|maybe|might be|seems like|sort of|kind of|'
        r'not sure|i guess|iirc|afaik|if i recall|i believe)\b', text, re.IGNORECASE))

    return {
        'voice_score': voice_score,
        'spec_score': spec_score,
        'vsd': vsd,
        'voice_gated': voice_score > 2.0,
        'casual_markers': casual_count,
        'misspellings': misspelling_count,
        'contractions': contractions,
        'em_dashes': em_dashes,
        'camel_cols': camel_cols,
        'filenames': filenames,
        'calcs': calcs,
        'tabs': tabs,
        'col_listings': col_listings,
        'hedges': hedges,
    }
