"""DNA-GPT Divergent Continuation Analysis via LLM API.

Truncates candidate text, regenerates continuations via LLM API,
measures n-gram overlap (BScore) between original and regenerated.
Ref: Yang et al. (2024) "DNA-GPT" (ICLR 2024)
"""

import re
import statistics
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _dna_ngrams(tokens, n):
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _dna_bscore(original_tokens, regenerated_tokens, ns=(2, 3, 4), weights=(0.25, 0.50, 0.25)):
    """Compute DNA-GPT BScore: weighted n-gram overlap."""
    scores = []
    for n, w in zip(ns, weights):
        orig_ng = set(_dna_ngrams(original_tokens, n))
        regen_ng = set(_dna_ngrams(regenerated_tokens, n))
        if not orig_ng or not regen_ng:
            scores.append(0.0)
            continue
        overlap = len(orig_ng & regen_ng)
        precision = overlap / len(regen_ng) if regen_ng else 0
        recall = overlap / len(orig_ng) if orig_ng else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores.append(f1 * w)
    return sum(scores)


def _dna_bscore_determination(bscore, bscore_max):
    """Map DNA-GPT BScore to (determination, confidence, reason) tuple."""
    if bscore >= 0.20 and bscore_max >= 0.22:
        det, conf = 'RED', min(0.90, 0.60 + bscore)
        reason = f"DNA-GPT: high continuation overlap (BScore={bscore:.3f}, max={bscore_max:.3f})"
    elif bscore >= 0.12:
        det, conf = 'AMBER', min(0.70, 0.40 + bscore)
        reason = f"DNA-GPT: elevated continuation overlap (BScore={bscore:.3f})"
    elif bscore >= 0.08:
        det, conf = 'YELLOW', min(0.40, 0.20 + bscore)
        reason = f"DNA-GPT: moderate continuation overlap (BScore={bscore:.3f})"
    else:
        det, conf = 'GREEN', max(0.0, 0.10 - bscore)
        reason = f"DNA-GPT: low continuation overlap (BScore={bscore:.3f}) -- likely human"
    return det, conf, reason


def _merge_multi_bscore_stability(full_result, bscores):
    """Attach multi-bscore stability fields to *full_result* in-place."""
    if len(bscores) >= 2:
        bscore_mean = statistics.mean(bscores)
        bscore_var = statistics.variance(bscores)
        stability = max(0.0, 1.0 - (bscore_var / 0.02))
    else:
        bscore_mean = bscores[0] if bscores else 0.0
        bscore_var = 0.0
        stability = 0.0

    full_result['multi_bscores'] = [round(b, 4) for b in bscores]
    full_result['bscore_variance'] = round(bscore_var, 6)
    full_result['bscore_stability'] = round(stability, 4)

    if stability >= 0.75 and bscore_mean >= 0.15:
        full_result['confidence'] = round(
            min(full_result.get('confidence', 0.0) + 0.10, 1.0), 4
        )


def _detect_text_format(text):
    """Detect structural features of the text for use in continuation prompts.

    Returns a list of human-readable feature descriptions, e.g.
    ``['paragraph breaks', 'numbered sections']``.
    """
    features = []
    if '\n\n' in text:
        features.append('paragraph breaks')
    if re.search(r'^\d+[\.\)]\s', text, re.M):
        features.append('numbered sections')
    if re.search(r'^[-•]\s', text, re.M):
        features.append('bullet points')
    if re.search(r'^#+\s', text, re.M):
        features.append('markdown headings')
    return features


def _format_hint_str(features):
    """Return a grammatically correct sentence fragment for *features*,
    or an empty string when there are no features.

    Examples::

        ['paragraph breaks']            ->  ' It uses paragraph breaks.'
        ['paragraph breaks', 'bullets'] ->  ' It uses paragraph breaks and bullets.'
    """
    if not features:
        return ""
    if len(features) == 1:
        return f" It uses {features[0]}."
    return f" It uses {', '.join(features[:-1])} and {features[-1]}."


def _dna_truncate_text(text, ratio=0.5):
    """Truncate text at the nearest structural boundary to *ratio*.

    Priority order:
    1. Paragraph break — snaps to the paragraph boundary closest to *ratio*
       (requires >= 3 paragraphs and that the continuation has >= 30 words).
    2. Sentence boundary — uses ``get_sentences()`` from text_utils (spaCy
       when available, robust regex fallback).
    3. Word-level fallback — used when < 4 sentences are detected.

    Returns (prefix, continuation).
    """
    # 1. Paragraph-level split
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]

    if len(paragraphs) >= 3:
        target_chars = int(len(text) * ratio)
        cumulative = 0
        best_cut_idx = None
        best_cut_dist = None
        for i, para in enumerate(paragraphs):
            cumulative += len(para) + 2  # +2 for the \n\n separator
            # Must keep at least one paragraph on each side
            if 1 <= i < len(paragraphs) - 1:
                dist = abs(cumulative - target_chars)
                if best_cut_idx is None or dist < best_cut_dist:
                    best_cut_idx = i + 1
                    best_cut_dist = dist
        if best_cut_idx is not None:
            prefix = '\n\n'.join(paragraphs[:best_cut_idx])
            continuation = '\n\n'.join(paragraphs[best_cut_idx:])
            if len(continuation.split()) >= 30:
                return prefix, continuation

    # 2. Sentence-level split
    from llm_detector.text_utils import get_sentences
    sentences = get_sentences(text)
    sentences = [s for s in sentences if len(s.strip()) > 5]

    if len(sentences) >= 4:
        cut = max(2, int(len(sentences) * ratio))
        prefix = ' '.join(sentences[:cut])
        continuation = ' '.join(sentences[cut:])
        if len(continuation.split()) >= 30:
            return prefix, continuation

    # 3. Word-level fallback
    words = text.split()
    mid = int(len(words) * ratio)
    return ' '.join(words[:mid]), ' '.join(words[mid:])


def _dna_call_anthropic(prefix, continuation_length, api_key,
                        model='claude-sonnet-4-20250514', n_samples=3, temperature=0.7):
    """Generate continuations using Anthropic API (parallel)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic  (required for continuation analysis with Anthropic)")
    client = anthropic.Anthropic(api_key=api_key)
    max_tokens = min(max(continuation_length * 2, 200), 4096)
    format_hints = _detect_text_format(prefix)
    format_str = _format_hint_str(format_hints)
    prompt = (
        "Continue the following text naturally, maintaining the same "
        f"style, tone, format, and topic.{format_str} "
        "Do not add any preamble or meta-commentary — "
        f"just continue writing:\n\n{prefix}"
    )

    def _call(_):
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else ""

    with ThreadPoolExecutor(max_workers=n_samples) as pool:
        continuations = list(pool.map(_call, range(n_samples)))
    return continuations


DNA_GPT_STORED_PROMPT_ID = 'pmpt_69a8ff3fd48081938b2de58954245ebf0f4f01733906fee0'


def _dna_call_openai(prefix, continuation_length, api_key,
                     model='gpt-4o-mini', n_samples=3, temperature=0.7):
    """Generate continuations using OpenAI Responses API (parallel)."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai  (required for continuation analysis with OpenAI)")
    client = openai.OpenAI(api_key=api_key)
    max_tokens = min(max(continuation_length * 2, 200), 4096)

    def _call(_):
        resp = client.responses.create(
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            instructions={
                "type": "stored_prompt",
                "id": DNA_GPT_STORED_PROMPT_ID,
            },
            input=prefix,
        )
        return resp.output_text or ""

    with ThreadPoolExecutor(max_workers=n_samples) as pool:
        continuations = list(pool.map(_call, range(n_samples)))
    return continuations


def run_continuation_api(text, api_key=None, provider='anthropic', model=None,
                         truncation_ratio=0.5, n_samples=3, temperature=0.7):
    """DNA-GPT divergent continuation analysis via LLM API."""
    word_count = len(text.split())

    if word_count < 150:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: insufficient text',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    if not api_key:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: no API key provided',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    prefix, original_continuation = _dna_truncate_text(text, truncation_ratio)
    if len(original_continuation.split()) < 30:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: continuation too short after truncation',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    orig_tokens = original_continuation.lower().split()
    continuation_word_count = len(orig_tokens)

    if model is None:
        model = 'claude-sonnet-4-20250514' if provider == 'anthropic' else 'gpt-4o-mini'

    try:
        if provider == 'anthropic':
            continuations = _dna_call_anthropic(prefix, continuation_word_count, api_key,
                                                model, n_samples, temperature)
        elif provider == 'openai':
            continuations = _dna_call_openai(prefix, continuation_word_count, api_key,
                                             model, n_samples, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")
    except Exception as e:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': f'DNA-GPT: API call failed ({e})',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    sample_scores = []
    for regen_text in continuations:
        regen_tokens = regen_text.lower().split()
        if len(regen_tokens) < 10:
            continue
        regen_tokens = regen_tokens[:int(len(orig_tokens) * 1.5)]
        bs = _dna_bscore(orig_tokens, regen_tokens)
        sample_scores.append(round(bs, 4))

    if not sample_scores:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: all regenerations failed or too short',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    bscore = statistics.mean(sample_scores)
    bscore_max = max(sample_scores)

    det, conf, reason = _dna_bscore_determination(bscore, bscore_max)

    return {
        'bscore': round(bscore, 4), 'bscore_max': round(bscore_max, 4),
        'bscore_samples': sample_scores, 'determination': det,
        'confidence': round(conf, 4), 'reason': reason,
        'n_samples': len(sample_scores), 'truncation_ratio': truncation_ratio,
        'continuation_words': continuation_word_count, 'word_count': word_count,
    }


def run_continuation_api_multi(text, api_key=None, provider='anthropic',
                                model=None, n_samples=3,
                                truncation_ratios=(0.3, 0.5, 0.7),
                                temperature=0.7):
    """Multi-truncation DNA-GPT continuation analysis via LLM API.

    Runs API-based continuation analysis at multiple truncation ratios and
    measures stability of the BScore across truncation points. High stability
    (low variance) across truncation points is an AI signal.
    """
    def _run_ratio(ratio):
        return ratio, run_continuation_api(
            text, api_key=api_key, provider=provider, model=model,
            truncation_ratio=ratio, n_samples=n_samples,
            temperature=temperature,
        )

    ratio_results = {}
    with ThreadPoolExecutor(max_workers=len(truncation_ratios)) as pool:
        for ratio, result in pool.map(lambda r: _run_ratio(r), truncation_ratios):
            ratio_results[ratio] = result

    bscores = [ratio_results[r].get('bscore', 0.0) for r in truncation_ratios]
    full_result = copy.deepcopy(ratio_results.get(0.5) or ratio_results[truncation_ratios[-1]])

    _merge_multi_bscore_stability(full_result, bscores)

    return full_result


# ── Anthropic Message Batches API ──────────────────────────────────────


def _prepare_batch_requests(texts, task_ids, model='claude-sonnet-4-20250514',
                            n_samples=3, truncation_ratios=(0.3, 0.5, 0.7),
                            temperature=0.7):
    """Pre-compute truncation prefixes and build batch request list.

    Returns (requests, metadata) where metadata maps custom_id back to
    (task_idx, ratio, sample_idx, orig_tokens, continuation_word_count).
    """
    requests = []
    metadata = {}
    skipped = {}  # task_idx -> reason

    for task_idx, (text, task_id) in enumerate(zip(texts, task_ids)):
        word_count = len(text.split())
        if word_count < 150:
            skipped[task_idx] = 'insufficient text'
            continue

        for ratio in truncation_ratios:
            prefix, original_continuation = _dna_truncate_text(text, ratio)
            if len(original_continuation.split()) < 30:
                continue

            orig_tokens = original_continuation.lower().split()
            continuation_word_count = len(orig_tokens)
            max_tokens = min(max(continuation_word_count * 2, 200), 4096)
            format_hints = _detect_text_format(prefix)
            format_str = _format_hint_str(format_hints)
            prompt = (
                "Continue the following text naturally, maintaining the same "
                f"style, tone, format, and topic.{format_str} "
                "Do not add any preamble or "
                "meta-commentary — just continue writing:\n\n" + prefix
            )

            for sample_idx in range(n_samples):
                custom_id = f"t{task_idx}_r{ratio}_s{sample_idx}"
                metadata[custom_id] = (
                    task_idx, ratio, sample_idx,
                    orig_tokens, continuation_word_count,
                )
                requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                })

    return requests, metadata, skipped


def _score_batch_results(raw_results, metadata, texts, task_ids,
                         n_samples=3, truncation_ratios=(0.3, 0.5, 0.7)):
    """Convert raw batch API results into per-task continuation dicts.

    Returns dict mapping task_idx -> continuation result (same format as
    run_continuation_api_multi).
    """
    # Group regenerated texts: task_idx -> ratio -> [texts]
    regen_map = {}
    for custom_id, regen_text in raw_results.items():
        if custom_id not in metadata:
            continue
        task_idx, ratio, sample_idx, orig_tokens, cont_wc = metadata[custom_id]
        regen_map.setdefault(task_idx, {}).setdefault(ratio, []).append(
            (regen_text, orig_tokens, cont_wc)
        )

    task_results = {}
    for task_idx, ratio_data in regen_map.items():
        text = texts[task_idx]
        word_count = len(text.split())
        ratio_results = {}

        for ratio, samples in ratio_data.items():
            if not samples:
                continue
            orig_tokens = samples[0][1]
            cont_wc = samples[0][2]

            sample_scores = []
            for regen_text, _, _ in samples:
                regen_tokens = regen_text.lower().split()
                if len(regen_tokens) < 10:
                    continue
                regen_tokens = regen_tokens[:int(len(orig_tokens) * 1.5)]
                bs = _dna_bscore(orig_tokens, regen_tokens)
                sample_scores.append(round(bs, 4))

            if not sample_scores:
                continue

            bscore = statistics.mean(sample_scores)
            bscore_max = max(sample_scores)

            det, conf, reason = _dna_bscore_determination(bscore, bscore_max)

            ratio_results[ratio] = {
                'bscore': round(bscore, 4), 'bscore_max': round(bscore_max, 4),
                'bscore_samples': sample_scores, 'determination': det,
                'confidence': round(conf, 4), 'reason': reason,
                'n_samples': len(sample_scores), 'truncation_ratio': ratio,
                'continuation_words': cont_wc, 'word_count': word_count,
            }

        if not ratio_results:
            continue

        # Merge multi-ratio results (same logic as run_continuation_api_multi)
        bscores = [ratio_results[r].get('bscore', 0.0)
                    for r in truncation_ratios if r in ratio_results]
        full_result = copy.deepcopy(
            ratio_results.get(0.5) or next(iter(ratio_results.values()))
        )

        _merge_multi_bscore_stability(full_result, bscores)

        task_results[task_idx] = full_result

    return task_results


def run_continuation_batch(texts, task_ids, api_key, model=None,
                           n_samples=3, truncation_ratios=(0.3, 0.5, 0.7),
                           temperature=0.7, poll_interval=10, progress_fn=None):
    """Run DNA-GPT continuation analysis for multiple texts via Anthropic Batches API.

    Args:
        texts: List of text strings to analyze.
        task_ids: List of task ID strings (same length as texts).
        api_key: Anthropic API key.
        model: Model name (default: claude-sonnet-4-20250514).
        n_samples: Regeneration samples per truncation ratio.
        truncation_ratios: Truncation ratios to test.
        temperature: Sampling temperature.
        poll_interval: Seconds between batch status polls.
        progress_fn: Optional callback(status_str) for progress updates.

    Returns:
        Dict mapping task_idx (int) -> continuation result dict.
        Tasks that were skipped (too short, etc.) won't have entries.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic  (required for batch continuation analysis)")

    if model is None:
        model = 'claude-sonnet-4-20250514'

    # 1. Prepare all requests
    requests, metadata, skipped = _prepare_batch_requests(
        texts, task_ids, model=model, n_samples=n_samples,
        truncation_ratios=truncation_ratios, temperature=temperature,
    )

    if not requests:
        return {}

    if progress_fn:
        progress_fn(f"Submitting batch of {len(requests)} API calls...")

    # 2. Submit batch
    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    if progress_fn:
        progress_fn(f"Batch {batch_id} submitted. Polling for completion...")

    # 3. Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        if progress_fn:
            counts = batch.request_counts
            progress_fn(
                f"Batch {batch_id}: {status} "
                f"(succeeded={counts.succeeded}, processing={counts.processing}, "
                f"errored={counts.errored})"
            )
        if status == 'ended':
            break
        time.sleep(poll_interval)

    # 4. Collect results
    raw_results = {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == 'succeeded':
            msg = entry.result.message
            text_out = msg.content[0].text if msg.content else ""
            raw_results[entry.custom_id] = text_out

    # 5. Score and merge
    return _score_batch_results(
        raw_results, metadata, texts, task_ids,
        n_samples=n_samples, truncation_ratios=truncation_ratios,
    )
