"""N-Gram Self-Similarity Index (NSSI) -- offline statistical fingerprinting.

Detects LLM-generated expository text via convergence of formulaic phrases,
power adjectives, discourse scaffolding, and information-theoretic signals.
"""

import re
import math
import zlib
import random
import statistics
from collections import Counter

# -- Formulaic Academic Phrases (pre-compiled) --
_FORMULAIC_PATTERNS_RAW = [
    (r'\bthis\s+(?:report|analysis|paper|study|section|document)\s+(?:provides?|presents?|examines?|dissects?|identifies?|evaluates?|proposes?|outlines?)\b', 1.5),
    (r'\b(?:it\s+is\s+(?:worth|important|imperative|crucial|essential|critical)\s+(?:noting|to\s+note|to\s+acknowledge|to\s+emphasize|to\s+recognize))\b', 2.0),
    (r'\b(?:to\s+address\s+this\s+(?:gap|issue|problem|challenge|limitation|deficiency|concern|shortcoming))\b', 1.5),
    (r'\b(?:perhaps\s+the\s+most\s+(?:\w+\s+)?(?:damning|significant|important|critical|notable|striking|concerning|alarming))\b', 2.0),
    (r'\b(?:(?:while|although)\s+(?:theoretically|conceptually|technically)\s+(?:sound|elegant|promising|valid|robust|appealing))\b', 2.0),
    (r'\b(?:the\s+(?:analysis|evidence|data|results?|findings?)\s+(?:suggests?|reveals?|indicates?|shows?|demonstrates?|confirms?)\s+that)\b', 1.0),
    (r'\b(?:this\s+(?:creates?|represents?|highlights?|underscores?|reveals?|illustrates?|exemplifies?)\s+(?:a|the|an))\b', 1.5),
    (r'\b(?:the\s+(?:primary|core|fundamental|critical|key|central|overarching)\s+(?:challenge|issue|problem|question|limitation|concern|insight|takeaway))\b', 1.0),
    (r'\b(?:in\s+(?:layman.s\s+terms|other\s+words|practical\s+terms|simple\s+terms|real.world\s+(?:terms|scenarios|situations)))\b', 1.5),
    (r'\b(?:defense\s+in\s+depth)\b', 1.0),
    (r'\b(?:arms?\s+race)\b', 0.5),
    (r'\b(?:the\s+(?:era|age|dawn)\s+of)\b', 0.5),
    (r'\b(?:a\s+(?:paradigm|fundamental|seismic|tectonic)\s+shift)\b', 2.0),
    (r'\b(?:the\s+(?:elephant|gorilla)\s+in\s+the\s+room)\b', 1.5),
    (r'\b(?:a\s+double.edged\s+sword)\b', 1.5),
    (r'\b(?:in\s+(?:conclusion|summary|closing),?)\b', 0.5),
    (r'\b(?:the\s+path\s+forward\s+(?:is|requires|demands|involves))\b', 1.5),
    (r'\b(?:(?:unless|until)\s+the\s+(?:community|industry|field|sector)\s+(?:adopts?|embraces?|commits?))\b', 2.0),
    (r'\b(?:the\s+(?:immediate|long.term|strategic)\s+(?:future|imperative|priority|solution)\s+(?:belongs?\s+to|lies?\s+in|requires?))\b', 2.0),
]

_FORMULAIC_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), weight)
    for pat, weight in _FORMULAIC_PATTERNS_RAW
]

# -- Power Adjectives --
_POWER_ADJ = re.compile(
    r'\b(?:comprehensive|exhaustive|rigorous|robust|holistic|systemic|'
    r'fundamental|critical|profound|decisive|catastrophic|perilous|'
    r'unprecedented|groundbreaking|transformative|paradigmatic|'
    r'monumental|pivotal|seminal|nascent|burgeoning|'
    r'overarching|multifaceted|nuanced|granular|bespoke|'
    r'actionable|scalable|tractable|non-trivial|intractable)\b',
    re.I
)

# -- Discourse Scaffolding --
_SCARE_QUOTE = re.compile(r'[\u201c\u201d][^\u201c\u201d]{2,40}[\u201c\u201d]|"[^"]{2,40}"')
_EM_DASH = re.compile(r'\u2014|--')
_PAREN = re.compile(r'\([^)]{12,}\)')
_COLON_EXPLAIN = re.compile(r':\s+[A-Z]')

# -- Demonstrative Monotony --
_DEMONSTRATIVE = re.compile(
    r'\bthis\s+(?:approach|method|framework|analysis|issue|mechanism|assumption|'
    r'limitation|strategy|technique|variant|disparity|metric|paradigm|'
    r'architecture|pipeline|deficiency|vulnerability|solution|concept|'
    r'pattern|signal|feature|constraint|observation|phenomenon|'
    r'suggests?|indicates?|creates?|ensures?|effectively|underscores?|'
    r'highlights?|represents?|reveals?|demonstrates?|means?|implies?|'
    r'raises?|poses?|necessitates?)\b', re.I
)

# -- Transition Connector Density --
_TRANSITION = re.compile(
    r'\b(?:however|furthermore|consequently|moreover|nevertheless|'
    r'additionally|specifically|crucially|ultimately|conversely|'
    r'notably|importantly|interestingly|remarkably|significantly|'
    r'simultaneously|correspondingly|paradoxically)\b', re.I
)

# -- Causal Reasoning Deficit --
_CAUSAL = re.compile(
    r'\b(?:because|since|'
    r'so\b(?!\s+(?:that|much|many|far|long|called))|'
    r'if|but|although|though|unless|whereas|'
    r'while(?=\s+\w+\s+(?:is|was|are|were|has|had|do|does|did|can|could|would|should|might|may))|'
    r'therefore|hence|thus|'
    r'think|believe|feel|know|suspect|doubt|wonder|guess|suppose|reckon|'
    r'maybe|perhaps|probably|apparently|presumably)\b', re.I
)


def _get_sentences(text):
    """Split text into sentences (regex fallback for NSSI)."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def run_self_similarity(text):
    """Compute N-Gram Self-Similarity Index (NSSI).

    Returns dict with individual feature scores and composite NSSI.
    """
    words = text.split()
    word_count = len(words)
    sentences = _get_sentences(text)
    n_sents = max(len(sentences), 1)

    if word_count < 200:
        return {
            'nssi_score': 0.0, 'nssi_signals': 0, 'nssi_active': [],
            'determination': None, 'confidence': 0.0,
            'reason': 'NSSI: text too short for analysis',
            'formulaic_density': 0.0, 'formulaic_weighted': 0.0,
            'power_adj_density': 0.0, 'scare_quote_density': 0.0,
            'emdash_density': 0.0, 'parenthetical_density': 0.0,
            'colon_density': 0.0, 'demonstrative_density': 0.0,
            'transition_density': 0.0, 'causal_density': 0.0,
            'causal_ratio': 0.0, 'this_the_start_rate': 0.0,
            'section_depth': 0, 'sent_length_cv': 0.0,
            'comp_ratio': 0.0, 'hapax_ratio': 0.0,
            'hapax_count': 0, 'unique_words': 0,
            'word_count': word_count, 'sentence_count': n_sents,
            'shuffled_comp_ratio': 0.0, 'structural_compression_delta': 0.0,
        }

    # 1. Formulaic phrase density
    formulaic_raw = 0
    formulaic_weighted = 0.0
    for compiled_pat, weight in _FORMULAIC_PATTERNS:
        hits = len(compiled_pat.findall(text))
        formulaic_raw += hits
        formulaic_weighted += hits * weight
    formulaic_density = formulaic_raw / n_sents
    formulaic_w_density = formulaic_weighted / n_sents

    # 2. Power adjective density
    power_hits = len(_POWER_ADJ.findall(text))
    power_density = power_hits / n_sents

    # 3. Discourse scaffolding
    scare_quotes = len(_SCARE_QUOTE.findall(text))
    emdashes = len(_EM_DASH.findall(text))
    parentheticals = len(_PAREN.findall(text))
    colon_explains = len(_COLON_EXPLAIN.findall(text))

    scare_density = scare_quotes / n_sents
    emdash_density = emdashes / n_sents
    paren_density = parentheticals / n_sents
    colon_density = colon_explains / n_sents

    # 4. Demonstrative monotony
    demo_hits = len(_DEMONSTRATIVE.findall(text))
    demo_density = demo_hits / n_sents

    # 5. Transition connector density
    trans_hits = len(_TRANSITION.findall(text))
    trans_density = trans_hits / n_sents

    # 5b. Causal reasoning deficit
    causal_hits = len(_CAUSAL.findall(text))
    causal_density = causal_hits / n_sents
    causal_ratio = (trans_hits + 1) / (causal_hits + 1)

    # 6. Sentence-start monotony
    starts = [s.split()[0].lower() for s in sentences if s.split()]
    this_the_starts = sum(1 for s in starts if s in ('this', 'the', 'these', 'those'))
    this_the_rate = this_the_starts / n_sents

    # 7. Section hierarchy depth
    headers = re.findall(r'^(\d+(?:\.\d+)*)\s+', text, re.M)
    section_depth = max((h.count('.') + 1 for h in headers), default=0)

    # 8. Sentence length CV
    sent_lens = [len(s.split()) for s in sentences]
    if len(sent_lens) > 2:
        sent_cv = statistics.stdev(sent_lens) / max(statistics.mean(sent_lens), 1)
    else:
        sent_cv = 0.5

    # -- Composite NSSI (12-signal convergence) --
    signals = []

    s1 = min(formulaic_w_density / 0.25, 1.0) if formulaic_w_density >= 0.04 else 0.0
    if s1 > 0: signals.append(('formulaic', s1))

    s2 = min(power_density / 0.30, 1.0) if power_density >= 0.08 else 0.0
    if s2 > 0: signals.append(('power_adj', s2))

    s3 = min(scare_density / 0.40, 1.0) if scare_density >= 0.08 else 0.0
    if s3 > 0: signals.append(('scare_quotes', s3))

    s4 = min(demo_density / 0.12, 1.0) if demo_density >= 0.03 else 0.0
    if s4 > 0: signals.append(('demonstratives', s4))

    s5 = min(trans_density / 0.20, 1.0) if trans_density >= 0.05 else 0.0
    if s5 > 0: signals.append(('transitions', s5))

    scaffold = emdash_density + paren_density + colon_density
    s6 = min(scaffold / 0.60, 1.0) if scaffold >= 0.15 else 0.0
    if s6 > 0: signals.append(('scaffolding', s6))

    s7 = min(this_the_rate / 0.35, 1.0) if this_the_rate >= 0.20 else 0.0
    if s7 > 0: signals.append(('start_monotony', s7))

    s8 = min(section_depth / 4.0, 1.0) if section_depth >= 3 else 0.0
    if s8 > 0: signals.append(('hierarchy', s8))

    s9 = 0.0
    if trans_hits >= 2 and causal_ratio >= 1.5:
        s9 = min((causal_ratio - 1.0) / 3.0, 1.0) * 0.5
    if s9 > 0: signals.append(('causal_deficit', s9))

    # s10: Burstiness
    s10 = 0.0
    if n_sents >= 4 and sent_cv <= 0.35:
        s10 = min((0.35 - sent_cv) / 0.15, 1.0)
    if s10 > 0: signals.append(('low_burstiness', s10))

    # s11: Zlib compression entropy
    text_bytes = text.encode('utf-8')
    original_len = max(len(text_bytes), 1)
    compressed_len = len(zlib.compress(text_bytes))
    comp_ratio = compressed_len / original_len

    s11 = 0.0
    if comp_ratio <= 0.42 and word_count >= 150:
        s11 = min((0.42 - comp_ratio) / 0.08, 1.0)
    if s11 > 0: signals.append(('high_compressibility', s11))

    # s12: Hapax legomena deficit
    clean_words = [w.strip('.,!?"\'():;').lower() for w in words]
    clean_words = [w for w in clean_words if w]
    word_freqs = Counter(clean_words)
    hapax_count = sum(1 for count in word_freqs.values() if count == 1)
    unique_words = len(word_freqs)
    hapax_ratio = hapax_count / unique_words if unique_words > 0 else 0.0

    s12 = 0.0
    if hapax_ratio <= 0.45 and word_count >= 150:
        s12 = min((0.45 - hapax_ratio) / 0.15, 1.0)
    if s12 > 0: signals.append(('hapax_deficit', s12))

    # s13: Structural compression delta (original vs shuffled) -- FEAT 5
    shuffled_words = list(clean_words)
    random.seed(42)
    random.shuffle(shuffled_words)
    shuffled_text = ' '.join(shuffled_words)
    shuffled_bytes = shuffled_text.encode('utf-8')
    shuffled_comp_len = len(zlib.compress(shuffled_bytes))
    shuffled_comp_ratio = shuffled_comp_len / max(len(shuffled_bytes), 1)
    structural_compression_delta = shuffled_comp_ratio - comp_ratio

    s13 = 0.0
    if structural_compression_delta < 0.03 and word_count >= 150:
        s13 = min((0.03 - structural_compression_delta) / 0.02, 1.0)
    if s13 > 0: signals.append(('low_structural_delta', s13))

    # -- Convergence scoring --
    n_active = len(signals)
    if n_active == 0:
        nssi_score = 0.0
    else:
        mean_strength = sum(s for _, s in signals) / n_active
        convergence = min(n_active / 5.5, 1.0)
        nssi_score = mean_strength * convergence
        if n_active >= 8:
            nssi_score = min(nssi_score * 1.3, 1.0)

    # -- Determination --
    if nssi_score >= 0.70 and n_active >= 7:
        det = 'RED'
        conf = min(0.85, nssi_score)
        reason = f"NSSI convergence (score={nssi_score:.2f}, {n_active} signals)"
    elif nssi_score >= 0.45 and n_active >= 5:
        det = 'AMBER'
        conf = min(0.65, nssi_score)
        reason = f"Elevated NSSI (score={nssi_score:.2f}, {n_active} signals)"
    elif nssi_score >= 0.25 and n_active >= 4:
        det = 'YELLOW'
        conf = min(0.40, nssi_score)
        reason = f"Moderate NSSI (score={nssi_score:.2f}, {n_active} signals)"
    else:
        det = None
        conf = 0.0
        reason = 'NSSI: insufficient signal convergence'

    return {
        'nssi_score': round(nssi_score, 4), 'nssi_signals': n_active,
        'nssi_active': [(name, round(val, 3)) for name, val in signals],
        'determination': det, 'confidence': round(conf, 4), 'reason': reason,
        'formulaic_density': round(formulaic_density, 4),
        'formulaic_weighted': round(formulaic_w_density, 4),
        'power_adj_density': round(power_density, 4),
        'scare_quote_density': round(scare_density, 4),
        'emdash_density': round(emdash_density, 4),
        'parenthetical_density': round(paren_density, 4),
        'colon_density': round(colon_density, 4),
        'demonstrative_density': round(demo_density, 4),
        'transition_density': round(trans_density, 4),
        'causal_density': round(causal_density, 4),
        'causal_ratio': round(causal_ratio, 4),
        'this_the_start_rate': round(this_the_rate, 4),
        'section_depth': section_depth,
        'sent_length_cv': round(sent_cv, 4),
        'comp_ratio': round(comp_ratio, 4),
        'hapax_ratio': round(hapax_ratio, 4),
        'hapax_count': hapax_count,
        'unique_words': unique_words,
        'word_count': word_count, 'sentence_count': n_sents,
        'shuffled_comp_ratio': round(shuffled_comp_ratio, 4),
        'structural_compression_delta': round(structural_compression_delta, 4),
    }
