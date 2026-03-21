"""Channel 3: Continuation-based detection (DNA-GPT / DNA-GPT-Local)."""

from llm_detector.channels import ChannelResult


def score_continuation(cont_result):
    """Score continuation channel. Returns ChannelResult."""
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []
    has_data = cont_result is not None

    if cont_result and cont_result.get('determination'):
        dna_det = cont_result['determination']
        bscore = cont_result.get('bscore', 0)
        sub['bscore'] = bscore

        proxy = cont_result.get('proxy_features')
        if proxy:
            sub['ncd'] = proxy.get('ncd', 0)
            sub['internal_overlap'] = proxy.get('internal_overlap', 0)
            sub['composite'] = proxy.get('composite', 0)
            label = 'Local'
        else:
            label = 'API'

        if dna_det == 'RED':
            score = min(0.95, cont_result.get('confidence', 0.65))
            severity = 'RED'
            parts.append(f"BScore={bscore:.3f}({label},RED)")
        elif dna_det == 'AMBER':
            score = min(0.70, cont_result.get('confidence', 0.50))
            severity = 'AMBER'
            parts.append(f"BScore={bscore:.3f}({label},AMBER)")
        elif dna_det == 'YELLOW':
            score = min(0.40, cont_result.get('confidence', 0.25))
            severity = 'YELLOW'
            parts.append(f"BScore={bscore:.3f}({label},YELLOW)")

    explanation = f"Continuation: {', '.join(parts)}" if parts else 'Continuation: no signals'

    return ChannelResult(
        'continuation', score, severity, explanation,
        mode_eligibility=['task_prompt', 'generic_aigt'],
        sub_signals=sub,
        data_sufficient=has_data,
    )
