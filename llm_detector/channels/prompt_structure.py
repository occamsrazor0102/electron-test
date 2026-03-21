"""Channel 1: Prompt-structure signals (task_prompt primary).

Combines preamble, prompt signature, voice dissonance, instruction density, SSI.
"""

from llm_detector.channels import ChannelResult


def score_prompt_structure(preamble_score, preamble_severity, prompt_sig, voice_dis, instr_density, word_count):
    """Score prompt-structure channel. Returns ChannelResult."""
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []

    # Preamble
    if preamble_severity == 'CRITICAL':
        return ChannelResult(
            'prompt_structure', 0.99, 'RED',
            'Preamble detection (critical hit)',
            mode_eligibility=['task_prompt', 'generic_aigt'],
            sub_signals={'preamble': 0.99},
        )
    if preamble_score >= 0.50:
        sub['preamble'] = preamble_score
        score = max(score, preamble_score)
        parts.append(f"preamble={preamble_score:.2f}")

    # Prompt signature
    comp = prompt_sig['composite']
    sub['prompt_signature'] = comp
    if comp >= 0.60:
        score = max(score, comp)
        severity = 'RED'
        parts.append(f"prompt_sig={comp:.2f}(RED)")
    elif comp >= 0.40:
        score = max(score, comp)
        severity = max(severity, 'AMBER', key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
        parts.append(f"prompt_sig={comp:.2f}(AMBER)")
    elif comp >= 0.20:
        score = max(score, comp * 0.7)
        severity = max(severity, 'YELLOW', key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
        parts.append(f"prompt_sig={comp:.2f}(YELLOW)")

    # Voice dissonance (voice-gated)
    if voice_dis['voice_gated']:
        vsd = voice_dis['vsd']
        sub['vsd_gated'] = vsd
        if vsd >= 50:
            score = max(score, 0.90)
            severity = 'RED'
            parts.append(f"VSD={vsd:.0f}(RED)")
        elif vsd >= 21:
            score = max(score, 0.70)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
                severity = 'AMBER'
            parts.append(f"VSD={vsd:.0f}(AMBER)")

    # Instruction density
    if instr_density:
        idi = instr_density['idi']
        sub['idi'] = idi
        if idi >= 12:
            score = max(score, 0.85)
            severity = 'RED'
            parts.append(f"IDI={idi:.0f}(RED)")
        elif idi >= 8:
            score = max(score, 0.65)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
                severity = 'AMBER'
            parts.append(f"IDI={idi:.0f}(AMBER)")

    # SSI (sterile specification)
    ssi_spec_threshold = 5.0 if voice_dis['contractions'] == 0 else 7.0
    ssi_triggered = (
        voice_dis['spec_score'] >= ssi_spec_threshold
        and voice_dis['voice_score'] < 0.5
        and voice_dis['hedges'] == 0
        and word_count >= 150
    )
    if ssi_triggered:
        sub['ssi'] = voice_dis['spec_score']
        if voice_dis['spec_score'] >= 8.0:
            score = max(score, 0.70)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
                severity = 'AMBER'
            parts.append(f"SSI={voice_dis['spec_score']:.0f}(AMBER)")
        else:
            score = max(score, 0.45)
            if ChannelResult.SEV_ORDER.get(severity, 0) < 1:
                severity = 'YELLOW'
            parts.append(f"SSI={voice_dis['spec_score']:.0f}(YELLOW)")

    # VSD ungated (very high)
    if not voice_dis['voice_gated'] and voice_dis['vsd'] >= 100:
        sub['vsd_ungated'] = voice_dis['vsd']
        score = max(score, 0.60)
        if ChannelResult.SEV_ORDER.get(severity, 0) < 2:
            severity = 'AMBER'
        parts.append(f"VSD_ungated={voice_dis['vsd']:.0f}")
    elif not voice_dis['voice_gated'] and voice_dis['vsd'] >= 21:
        sub['vsd_ungated'] = voice_dis['vsd']
        score = max(score, 0.30)
        if ChannelResult.SEV_ORDER.get(severity, 0) < 1:
            severity = 'YELLOW'

    explanation = f"Prompt-structure: {', '.join(parts)}" if parts else 'Prompt-structure: no signals'

    return ChannelResult(
        'prompt_structure', score, severity, explanation,
        mode_eligibility=['task_prompt', 'generic_aigt'],
        sub_signals=sub,
    )
