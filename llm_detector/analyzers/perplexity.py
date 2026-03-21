"""Local perplexity scoring via distilgpt2.

AI text has low perplexity (< 20); human text typically > 35.
Ref: GLTR (Gehrmann et al. 2019), DetectGPT (Mitchell et al. 2023)

Surprisal diversity features based on:
- DivEye (variance of per-token surprisal)
- "When AI Settles Down" (volatility decay across text halves)
"""

import logging
import statistics as _statistics
import zlib

from llm_detector.compat import HAS_PERPLEXITY, get_perplexity_model, HAS_BINOCULARS, get_binoculars_model
from llm_detector.text_utils import get_sentences

logger = logging.getLogger(__name__)

if HAS_PERPLEXITY:
    import torch as _torch

_PPL_EMPTY = {
    'perplexity': 0.0, 'determination': None, 'confidence': 0.0,
    'surprisal_variance': 0.0, 'surprisal_first_half_var': 0.0,
    'surprisal_second_half_var': 0.0, 'volatility_decay_ratio': 1.0,
    'comp_ratio': 0.0, 'zlib_normalized_ppl': 0.0, 'comp_ppl_ratio': 0.0,
    'token_losses': None,
    'binoculars_score': 0.0, 'binoculars_determination': None,
    'ppl_burstiness': 0.0, 'sentence_ppl_count': 0,
}


def run_perplexity(text, model_id=None):
    """Calculate token-level perplexity using a causal language model.

    Args:
        text: Input text.
        model_id: HuggingFace model identifier (default: Qwen2.5-0.5B).

    Returns dict with perplexity, determination, confidence, and
    surprisal diversity features (variance, half-variances, decay ratio).
    """
    if not HAS_PERPLEXITY:
        return {**_PPL_EMPTY, 'reason': 'Perplexity scoring unavailable (transformers/torch not installed)'}

    words = text.split()
    if len(words) < 50:
        return {**_PPL_EMPTY, 'reason': 'Perplexity: text too short'}

    model, tokenizer = get_perplexity_model(model_id)

    encodings = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=1024)
    input_ids = encodings.input_ids

    if input_ids.size(1) < 10:
        return {**_PPL_EMPTY, 'reason': 'Perplexity: too few tokens after encoding'}

    with _torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = _torch.exp(loss).item()

    # ── Binoculars: contrastive cross-model ratio ──
    # Ref: Hans et al. (2024) "Spotting LLMs With Binoculars"
    # Low ratio (performer/observer) indicates AI-generated text.
    binoculars_score = 0.0
    binoculars_det = None
    if HAS_BINOCULARS:
        try:
            observer, obs_tokenizer = get_binoculars_model()
            if observer is not None:
                # Re-tokenize with observer's tokenizer (may differ from primary model)
                obs_enc = obs_tokenizer(text, return_tensors='pt', truncation=True,
                                        max_length=1024)
                obs_ids = obs_enc.input_ids
                with _torch.no_grad():
                    obs_outputs = observer(obs_ids, labels=obs_ids)
                    obs_ppl = _torch.exp(obs_outputs.loss).item()
                binoculars_score = round(ppl / max(obs_ppl, 1e-6), 4)
                if binoculars_score < 0.70:
                    binoculars_det = 'AMBER'
                elif binoculars_score < 0.85:
                    binoculars_det = 'YELLOW'
        except Exception as exc:
            logger.debug("Binoculars scoring failed (supplementary): %s", exc)

    # ── FEAT 7: Compression-perplexity divergence ──
    text_bytes = text.encode('utf-8')
    comp_len = len(zlib.compress(text_bytes))
    comp_ratio = comp_len / max(len(text_bytes), 1)
    zlib_normalized_ppl = ppl * comp_ratio
    comp_ppl_ratio = comp_ratio / max(ppl / 100.0, 0.01)

    # ── Surprisal diversity & volatility decay ──
    surprisal_variance = 0.0
    volatility_decay_ratio = 1.0
    first_half_var = 0.0
    second_half_var = 0.0
    n_tokens = 0
    token_losses_list = None
    try:
        with _torch.no_grad():
            logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fn = _torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        losses = per_token_loss.float().cpu().numpy()
        n_tokens = len(losses)
        token_losses_list = losses.tolist()

        if n_tokens >= 10:
            surprisal_variance = float(losses.var())
            mid = n_tokens // 2
            first_half_var = float(losses[:mid].var()) if mid > 1 else surprisal_variance
            second_half_var = float(losses[mid:].var()) if (n_tokens - mid) > 1 else surprisal_variance
            volatility_decay_ratio = (first_half_var / second_half_var) if second_half_var > 1e-6 else 1.0
    except Exception as exc:
        logger.debug("Surprisal variance features failed (supplementary): %s", exc)

    # ── Perplexity burstiness: per-sentence PPL variance ──
    # Humans write in bursts — complex sentences followed by simple ones.
    # LLMs maintain flat, uniform perplexity across sentences.
    ppl_burstiness = 0.0
    sentence_ppl_count = 0
    if token_losses_list and n_tokens >= 20:
        try:
            sentences = get_sentences(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) >= 3:
                # Approximate sentence boundaries in token space by tokenizing
                # each sentence and mapping token counts
                sentence_token_counts = []
                for sent in sentences:
                    sent_enc = tokenizer(sent, return_tensors='pt', truncation=True, max_length=512)
                    # Subtract 1 for BOS/special tokens if present, but ensure >= 1
                    n_tok = max(sent_enc.input_ids.size(1) - 1, 1)
                    sentence_token_counts.append(n_tok)

                # Map per-token losses to sentences
                sentence_mean_losses = []
                offset = 0
                for stc in sentence_token_counts:
                    end = min(offset + stc, len(token_losses_list))
                    if end > offset:
                        chunk = token_losses_list[offset:end]
                        sentence_mean_losses.append(_statistics.mean(chunk))
                    offset = end
                    if offset >= len(token_losses_list):
                        break

                if len(sentence_mean_losses) >= 3:
                    ppl_burstiness = _statistics.variance(sentence_mean_losses)
                    sentence_ppl_count = len(sentence_mean_losses)
        except Exception as exc:
            logger.debug("Perplexity burstiness failed (supplementary): %s", exc)

    if ppl <= 15.0:
        det = 'AMBER'
        conf = min(0.65, (20.0 - ppl) / 20.0)
        reason = f"Low perplexity ({ppl:.1f}): highly predictable text"
    elif ppl <= 25.0:
        det = 'YELLOW'
        conf = min(0.35, (30.0 - ppl) / 30.0)
        reason = f"Moderate perplexity ({ppl:.1f}): somewhat predictable"
    else:
        det = None
        conf = 0.0
        reason = f"Normal perplexity ({ppl:.1f}): consistent with human text"

    # DivEye + Volatility compound upgrade (Basani & Chen; Sun et al.)
    diveye_signal = surprisal_variance < 2.0 and n_tokens >= 30
    volatility_signal = volatility_decay_ratio > 1.5 and n_tokens >= 40

    if diveye_signal and volatility_signal:
        if det is None:
            det = 'YELLOW'
            conf = min(0.40, 0.20 + (2.0 - surprisal_variance) * 0.05
                       + (volatility_decay_ratio - 1.0) * 0.05)
            reason = (f"Surprisal uniformity (var={surprisal_variance:.2f}, "
                      f"decay={volatility_decay_ratio:.2f}): machine rhythm detected")
        elif det == 'YELLOW':
            det = 'AMBER'
            conf = min(0.65, conf + 0.15)
            reason += (f" + DivEye(var={surprisal_variance:.2f}, "
                       f"decay={volatility_decay_ratio:.2f})")
        elif det == 'AMBER':
            conf = min(0.80, conf + 0.10)
            reason += (f" + DivEye(var={surprisal_variance:.2f}, "
                       f"decay={volatility_decay_ratio:.2f})")
    elif diveye_signal or volatility_signal:
        if det is not None:
            conf = min(conf + 0.05, 0.70)
            if diveye_signal:
                reason += f" + low_variance({surprisal_variance:.2f})"
            else:
                reason += f" + volatility_decay({volatility_decay_ratio:.2f})"

    # FEAT 7: Zlib-normalized perplexity compound signal
    zlib_ppl_signal = zlib_normalized_ppl < 8.0 and n_tokens >= 30
    if zlib_ppl_signal:
        if det is None:
            det = 'YELLOW'
            conf = min(0.35, 0.15 + (8.0 - zlib_normalized_ppl) * 0.02)
            reason = f"Zlib-normalized PPL ({zlib_normalized_ppl:.1f}): predictable and compressible"
        elif det in ('YELLOW', 'AMBER'):
            conf = min(conf + 0.05, 0.80)
            reason += f" + zlib_ppl({zlib_normalized_ppl:.1f})"

    return {
        'perplexity': round(ppl, 2),
        'determination': det,
        'confidence': conf,
        'reason': reason,
        'surprisal_variance': round(surprisal_variance, 4),
        'surprisal_first_half_var': round(first_half_var, 4),
        'surprisal_second_half_var': round(second_half_var, 4),
        'volatility_decay_ratio': round(volatility_decay_ratio, 4),
        'comp_ratio': round(comp_ratio, 4),
        'zlib_normalized_ppl': round(zlib_normalized_ppl, 2),
        'comp_ppl_ratio': round(comp_ppl_ratio, 4),
        'token_losses': token_losses_list,
        'binoculars_score': binoculars_score,
        'binoculars_determination': binoculars_det,
        'ppl_burstiness': round(ppl_burstiness, 4),
        'sentence_ppl_count': sentence_ppl_count,
    }
