"""DNA-GPT Proxy -- zero-LLM divergent continuation analysis.

Uses a backoff n-gram language model as surrogate for LLM regeneration.
Ref: Yang et al. (2024) "DNA-GPT" (ICLR 2024)
Ref: Li et al. (2004) "The Similarity Metric" (NCD theory)
"""

import re
import math
import zlib
import statistics
from collections import Counter, defaultdict

from llm_detector.analyzers.continuation_api import _dna_ngrams, _dna_bscore, _dna_truncate_text
from llm_detector.text_utils import type_token_ratio

_TOKEN_RE = re.compile(r'\w+|[^\w\s]')


def _proxy_tokenize(text):
    """Tokenize for n-gram LM. Returns lowercased word/punct tokens."""
    return _TOKEN_RE.findall(text.lower())


class _BackoffNGramLM:
    """Simple backoff n-gram language model for DNA-GPT proxy regeneration."""

    def __init__(self, order=5, alpha=0.1):
        self.order = max(order, 1)
        self.alpha = alpha
        self.tables = [defaultdict(Counter) for _ in range(self.order)]
        self.vocab = set()

    def fit(self, texts):
        """Train on an iterable of text strings."""
        bos = ['<s>'] * (self.order - 1)
        for text in texts:
            toks = bos + _proxy_tokenize(text) + ['</s>']
            self.vocab.update(toks)
            for i in range(self.order - 1, len(toks)):
                for ctx_len in range(self.order):
                    ctx = tuple(toks[i - ctx_len:i]) if ctx_len else ()
                    self.tables[ctx_len][ctx][toks[i]] += 1

    def _counts(self, context):
        """Get counts for context with backoff."""
        max_ctx = min(len(context), self.order - 1)
        for ctx_len in range(max_ctx, -1, -1):
            ctx = tuple(context[-ctx_len:]) if ctx_len else ()
            counts = self.tables[ctx_len].get(ctx)
            if counts:
                return counts
        return Counter({t: 1 for t in self.vocab}) if self.vocab else Counter({'</s>': 1})

    def sample_next(self, context):
        """Sample a single next token given context."""
        import random as _random
        counts = self._counts(context)
        items = list(counts.items())
        total = sum(c for _, c in items)
        r = _random.random() * total
        acc = 0.0
        for tok, c in items:
            acc += c
            if acc >= r:
                return tok
        return items[-1][0]

    def logprob(self, token, context):
        """Log-probability of token given context (with Laplace smoothing)."""
        counts = self._counts(context)
        total = sum(counts.values())
        vocab_size = max(len(self.vocab), 1)
        p = (counts.get(token, 0) + self.alpha) / (total + self.alpha * vocab_size)
        return math.log(p)

    def sample_suffix(self, prefix_tokens, length):
        """Generate a continuation of `length` tokens from prefix context."""
        ctx = ['<s>'] * (self.order - 1) + list(prefix_tokens)
        out = []
        for _ in range(length):
            tok = self.sample_next(ctx)
            if tok == '</s>':
                break
            out.append(tok)
            ctx.append(tok)
        return out


def _calculate_ncd(prefix, suffix):
    """Normalized Compression Distance between prefix and suffix."""
    x = prefix.encode('utf-8')
    y = suffix.encode('utf-8')
    xy = x + b' ' + y

    c_x = len(zlib.compress(x))
    c_y = len(zlib.compress(y))
    c_xy = len(zlib.compress(xy))

    denom = max(c_x, c_y)
    if denom == 0:
        return 0.0
    return (c_xy - min(c_x, c_y)) / denom


def _internal_ngram_overlap(prefix_tokens, suffix_tokens, ns=(3, 4)):
    """Fraction of suffix n-grams that appear in prefix (echo effect)."""
    if not suffix_tokens:
        return 0.0

    total_weight = 0.0
    weighted_overlap = 0.0

    for n in ns:
        pfx_ng = set(_dna_ngrams(prefix_tokens, n))
        sfx_ng = set(_dna_ngrams(suffix_tokens, n))
        if not sfx_ng:
            continue
        w = n * math.log(n) if n > 1 else 1.0
        overlap = len(pfx_ng & sfx_ng) / len(sfx_ng)
        weighted_overlap += w * overlap
        total_weight += w

    return weighted_overlap / total_weight if total_weight else 0.0


def _repeated_ngram_rate(tokens, n=4):
    """Fraction of n-grams that are repetitions (monotonicity signal)."""
    count = max(0, len(tokens) - n + 1)
    if count == 0:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(count)]
    return 1.0 - len(set(grams)) / len(grams)


def _conditional_surprisal(lm, prefix_tokens, suffix_tokens):
    """Mean negative log-probability of suffix given prefix under LM."""
    ctx = ['<s>'] * (lm.order - 1) + list(prefix_tokens)
    total = 0.0
    for tok in suffix_tokens:
        total -= lm.logprob(tok, ctx)
        ctx.append(tok)
    return total / max(1, len(suffix_tokens))


def _surprisal_improvement_curve(lm, full_tokens, splits=(0.25, 0.50, 0.75)):
    """Measure how conditional surprisal changes with increasing prefix length.

    Returns dict with surprisal at each split point and the improvement rate.
    """
    n = len(full_tokens)
    if n < 40:
        return {'surprisal_curve': [], 'improvement_rate': 0.0}

    tail_start = int(n * 0.75)
    tail_tokens = full_tokens[tail_start:]

    surprisals = []
    for split in splits:
        prefix_end = int(n * split)
        if prefix_end < 10:
            continue
        prefix = full_tokens[:prefix_end]
        surp = _conditional_surprisal(lm, prefix, tail_tokens)
        surprisals.append((split, round(surp, 4)))

    if len(surprisals) >= 2:
        first = surprisals[0][1]
        last = surprisals[-1][1]
        improvement_rate = (first - last) / max(first, 1e-6)
    else:
        improvement_rate = 0.0

    return {
        'surprisal_curve': surprisals,
        'improvement_rate': round(improvement_rate, 4),
    }


def _multi_segment_ncd(text, n_segments=4):
    """Compute NCD between all pairs of text segments.

    Low variance = all segments are similarly redundant = AI signal.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < n_segments * 2:
        return {'ncd_mean': 0.0, 'ncd_variance': 0.0, 'ncd_min': 0.0, 'n_pairs': 0}

    seg_size = len(sentences) // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_size
        end = start + seg_size if i < n_segments - 1 else len(sentences)
        segments.append(' '.join(sentences[start:end]))

    ncds = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            ncd = _calculate_ncd(segments[i], segments[j])
            ncds.append(ncd)

    if not ncds:
        return {'ncd_mean': 0.0, 'ncd_variance': 0.0, 'ncd_min': 0.0, 'n_pairs': 0}

    return {
        'ncd_mean': round(statistics.mean(ncds), 4),
        'ncd_variance': round(statistics.variance(ncds) if len(ncds) >= 2 else 0.0, 6),
        'ncd_min': round(min(ncds), 4),
        'n_pairs': len(ncds),
    }


def run_continuation_local(text, gamma=0.5, K=32, order=5):
    """Zero-LLM DNA-GPT proxy via backoff n-gram language model."""
    word_count = len(text.split())

    if word_count < 80:
        return {
            'bscore': 0.0, 'bscore_samples': [], 'determination': None,
            'confidence': 0.0, 'reason': 'DNA-GPT-Local: insufficient text (<80 words)',
            'n_samples': 0, 'truncation_ratio': gamma, 'word_count': word_count,
            'proxy_features': {},
        }

    prefix_text, suffix_text = _dna_truncate_text(text, gamma)
    prefix_tokens = _proxy_tokenize(prefix_text)
    suffix_tokens = _proxy_tokenize(suffix_text)

    if len(suffix_tokens) < 20:
        return {
            'bscore': 0.0, 'bscore_samples': [], 'determination': None,
            'confidence': 0.0, 'reason': 'DNA-GPT-Local: suffix too short after split',
            'n_samples': 0, 'truncation_ratio': gamma, 'word_count': word_count,
            'proxy_features': {},
        }

    lm = _BackoffNGramLM(order=order)
    lm.fit([prefix_text])

    sample_scores = []
    for _ in range(K):
        regen = lm.sample_suffix(prefix_tokens, len(suffix_tokens))
        if len(regen) < 10:
            continue
        bs = _dna_bscore(suffix_tokens, regen)
        sample_scores.append(round(bs, 4))

    if not sample_scores:
        sample_scores = [0.0]

    bscore = statistics.mean(sample_scores)
    bscore_max = max(sample_scores)

    ncd = _calculate_ncd(prefix_text, suffix_text)
    internal_overlap = _internal_ngram_overlap(prefix_tokens, suffix_tokens)
    cond_surp = _conditional_surprisal(lm, prefix_tokens, suffix_tokens)
    repeat4 = _repeated_ngram_rate(suffix_tokens, 4)
    ttr = type_token_ratio(suffix_tokens)

    # FEAT 2: Cross-prefix surprisal improvement curve
    all_tokens = _proxy_tokenize(text)
    surp_curve = _surprisal_improvement_curve(lm, all_tokens)

    proxy_features = {
        'ncd': round(ncd, 4),
        'internal_overlap': round(internal_overlap, 4),
        'cond_surprisal': round(cond_surp, 4),
        'repeat4': round(repeat4, 4),
        'ttr': round(ttr, 4),
        'surprisal_curve': surp_curve['surprisal_curve'],
        'improvement_rate': surp_curve['improvement_rate'],
    }

    # FEAT 6: Multi-segment NCD matrix
    multi_ncd = _multi_segment_ncd(text)
    proxy_features['ncd_matrix_mean'] = multi_ncd['ncd_mean']
    proxy_features['ncd_matrix_variance'] = multi_ncd['ncd_variance']
    proxy_features['ncd_matrix_min'] = multi_ncd['ncd_min']

    # Composite scoring
    ncd_signal = max(0.0, (1.0 - ncd) / 0.15)
    overlap_signal = max(0.0, min(1.0, (internal_overlap - 0.05) / 0.30))
    repeat_signal = max(0.0, min(1.0, repeat4 / 0.15))
    ttr_signal = max(0.0, min(1.0, (0.55 - ttr) / 0.20))
    bscore_signal = min(1.0, bscore / 0.15)

    # NCD uniformity signal (FEAT 6)
    ncd_uniformity_signal = 0.0
    if multi_ncd['ncd_variance'] < 0.002 and multi_ncd['n_pairs'] >= 3:
        ncd_uniformity_signal = min((0.002 - multi_ncd['ncd_variance']) / 0.001, 1.0) * 0.5

    composite = (
        0.30 * bscore_signal +
        0.25 * ncd_signal +
        0.20 * overlap_signal +
        0.10 * repeat_signal +
        0.10 * ttr_signal +
        0.05 * max(0.0, min(1.0, (5.0 - cond_surp) / 3.0))
    )
    # Blend in NCD uniformity with small weight
    if ncd_uniformity_signal > 0:
        composite = composite * 0.93 + ncd_uniformity_signal * 0.07

    proxy_features['composite'] = round(composite, 4)

    if composite >= 0.60 and (ncd_signal >= 0.4 or overlap_signal >= 0.5):
        det = 'RED'
        conf = min(0.95, 0.55 + composite * 0.40)
        reason = (f"DNA-GPT-Local: high self-consistency "
                  f"(composite={composite:.2f}, NCD={ncd:.3f}, "
                  f"overlap={internal_overlap:.3f})")
    elif composite >= 0.40:
        det = 'AMBER'
        conf = min(0.70, 0.35 + composite * 0.40)
        reason = (f"DNA-GPT-Local: elevated predictability "
                  f"(composite={composite:.2f}, NCD={ncd:.3f})")
    elif composite >= 0.25:
        det = 'YELLOW'
        conf = min(0.40, 0.15 + composite * 0.35)
        reason = (f"DNA-GPT-Local: moderate self-consistency "
                  f"(composite={composite:.2f})")
    else:
        det = None
        conf = 0.0
        reason = (f"DNA-GPT-Local: low predictability "
                  f"(composite={composite:.2f}) -- likely human")

    return {
        'bscore': round(bscore, 4),
        'bscore_max': round(bscore_max, 4),
        'bscore_samples': sample_scores,
        'determination': det,
        'confidence': round(conf, 4),
        'reason': reason,
        'n_samples': len(sample_scores),
        'truncation_ratio': gamma,
        'continuation_words': len(suffix_tokens),
        'word_count': word_count,
        'proxy_features': proxy_features,
    }


def run_continuation_local_multi(text, gammas=(0.3, 0.5, 0.7), K=16, order=5):
    """Multi-truncation DNA-GPT local proxy.

    Runs continuation analysis at multiple truncation ratios and measures
    stability of the composite score. High stability (low variance) across
    truncation points is an AI signal.
    """
    composites = []
    full_result = None

    for gamma in gammas:
        result = run_continuation_local(text, gamma=gamma, K=K, order=order)
        comp = result.get('proxy_features', {}).get('composite', 0.0)
        composites.append(comp)
        if gamma == 0.5:
            full_result = result

    if full_result is None:
        full_result = result

    if len(composites) >= 2:
        comp_mean = statistics.mean(composites)
        comp_var = statistics.variance(composites)
        stability = max(0.0, 1.0 - (comp_var / 0.08))
    else:
        comp_mean = composites[0] if composites else 0.0
        comp_var = 0.0
        stability = 0.0

    full_result['proxy_features']['multi_composites'] = [round(c, 4) for c in composites]
    full_result['proxy_features']['composite_variance'] = round(comp_var, 6)
    full_result['proxy_features']['composite_stability'] = round(stability, 4)

    # Stability boosts the composite when it agrees with the primary signal
    if stability >= 0.75 and comp_mean >= 0.30:
        boosted = min(full_result['proxy_features']['composite'] + 0.10, 1.0)
        full_result['proxy_features']['composite'] = round(boosted, 4)

    return full_result
