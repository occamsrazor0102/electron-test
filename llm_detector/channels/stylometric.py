"""Channel 2: Stylometric signals (generic_aigt primary).

Combines NSSI, semantic resonance, perplexity, and fingerprints.
"""

from llm_detector.channels import ChannelResult


def score_stylometric(fingerprint_score, self_sim, voice_dis=None, semantic=None, ppl=None, tocsin=None, semantic_flow=None):
    """Score stylometric channel. Returns ChannelResult."""
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []

    # Fingerprints: supporting-only
    if fingerprint_score > 0:
        sub['fingerprints'] = fingerprint_score

    # NSSI: primary stylometric signal
    if self_sim and self_sim.get('determination'):
        nssi_det = self_sim['determination']
        nssi_score = self_sim.get('nssi_score', 0)
        nssi_signals = self_sim.get('nssi_signals', 0)
        sub['nssi_score'] = nssi_score
        sub['nssi_signals'] = nssi_signals

        if nssi_det == 'RED':
            score = max(score, min(0.85, self_sim.get('confidence', 0.80)))
            severity = 'RED'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(RED)")
        elif nssi_det == 'AMBER':
            score = max(score, min(0.65, self_sim.get('confidence', 0.60)))
            severity = 'AMBER'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(AMBER)")
        elif nssi_det == 'YELLOW':
            score = max(score, min(0.40, self_sim.get('confidence', 0.30)))
            severity = 'YELLOW'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(YELLOW)")

    # Semantic resonance: supporting signal
    if semantic and semantic.get('determination'):
        sem_det = semantic['determination']
        sem_delta = semantic.get('semantic_delta', 0)
        sub['semantic_ai_score'] = semantic.get('semantic_ai_mean', 0)
        sub['semantic_delta'] = sem_delta

        if sem_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"Sem=AMBER(delta={sem_delta:.2f},boost)")
            else:
                score = max(score, semantic.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"Sem=AMBER(delta={sem_delta:.2f})")
        elif sem_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"Sem=YELLOW(delta={sem_delta:.2f},supporting)")
            else:
                score = max(score, semantic.get('confidence', 0.30))
                severity = 'YELLOW'
                parts.append(f"Sem=YELLOW(delta={sem_delta:.2f})")

    # Perplexity: supporting signal
    if ppl and ppl.get('determination'):
        ppl_det = ppl['determination']
        ppl_val = ppl.get('perplexity', 0)
        sub['perplexity'] = ppl_val

        if ppl_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"PPL={ppl_val:.0f}(AMBER,boost)")
            else:
                score = max(score, ppl.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"PPL={ppl_val:.0f}(AMBER)")
        elif ppl_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"PPL={ppl_val:.0f}(YELLOW,supporting)")
            else:
                score = max(score, ppl.get('confidence', 0.30))
                severity = 'YELLOW'
                parts.append(f"PPL={ppl_val:.0f}(YELLOW)")

    # Surprisal variance & volatility decay: supporting signal
    # Low variance + high decay ratio signals AI-generated uniformity
    if ppl:
        s_var = ppl.get('surprisal_variance', 0.0)
        v_decay = ppl.get('volatility_decay_ratio', 1.0)
        if s_var > 0:
            sub['surprisal_variance'] = s_var
            sub['volatility_decay_ratio'] = v_decay
            if severity != 'GREEN' and s_var > 0:
                if s_var < 2.0 and v_decay > 1.5:
                    score = min(score + 0.08, 1.0)
                    parts.append(f"SurpVar={s_var:.2f}/Decay={v_decay:.2f}(boost)")
                elif s_var < 3.0 and v_decay > 1.2:
                    score = min(score + 0.04, 1.0)
                    parts.append(f"SurpVar={s_var:.2f}/Decay={v_decay:.2f}(mild)")

    # Token cohesiveness (TOCSIN): supporting signal
    if tocsin and tocsin.get('determination'):
        tocsin_det = tocsin['determination']
        sub['cohesiveness'] = tocsin.get('cohesiveness', 0)

        if tocsin_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"TOCSIN=AMBER(coh={tocsin['cohesiveness']:.4f},boost)")
            else:
                score = max(score, tocsin.get('confidence', 0.50))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"TOCSIN=AMBER(coh={tocsin['cohesiveness']:.4f})")
        elif tocsin_det == 'YELLOW' and severity != 'GREEN':
            score = min(score + 0.05, 1.0)
            parts.append(f"TOCSIN=YELLOW(supporting)")

    # Binoculars contrastive LM ratio: supporting signal
    if ppl and ppl.get('binoculars_determination'):
        bino_det = ppl['binoculars_determination']
        bino_score = ppl.get('binoculars_score', 0)
        sub['binoculars_score'] = bino_score

        if bino_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"Bino={bino_score:.2f}(AMBER,boost)")
            else:
                score = max(score, ppl.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"Bino={bino_score:.2f}(AMBER)")
        elif bino_det == 'YELLOW' and severity != 'GREEN':
            score = min(score + 0.05, 1.0)
            parts.append(f"Bino={bino_score:.2f}(YELLOW,supporting)")

    # Semantic flow: inter-sentence smoothness signal
    if semantic_flow and semantic_flow.get('determination'):
        flow_det = semantic_flow['determination']
        flow_var = semantic_flow.get('flow_variance', 0)
        sub['flow_variance'] = flow_var

        if flow_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"Flow=AMBER(var={flow_var:.4f},boost)")
            else:
                score = max(score, semantic_flow.get('confidence', 0.50))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"Flow=AMBER(var={flow_var:.4f})")
        elif flow_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"Flow=YELLOW(var={flow_var:.4f},supporting)")
            else:
                score = max(score, semantic_flow.get('confidence', 0.25))
                severity = 'YELLOW'
                parts.append(f"Flow=YELLOW(var={flow_var:.4f})")

    # Perplexity burstiness: low burstiness = AI uniform rhythm
    if ppl and ppl.get('ppl_burstiness', 0) > 0 and ppl.get('sentence_ppl_count', 0) >= 3:
        burst = ppl['ppl_burstiness']
        sub['ppl_burstiness'] = burst
        if severity != 'GREEN' and burst < 0.5:
            score = min(score + 0.06, 1.0)
            parts.append(f"Burst={burst:.2f}(low,supporting)")

    # Fingerprints add supporting weight if any stylometric signal is active
    if fingerprint_score >= 0.30 and severity != 'GREEN':
        score = min(score + 0.10, 1.0)
        parts.append(f"fingerprint={fingerprint_score:.2f}(supporting)")

    explanation = f"Stylometry: {', '.join(parts)}" if parts else 'Stylometry: no signals'

    # Mark data_sufficient=False when no analyzers could run
    # (no NSSI, no semantic, no perplexity, no TOCSIN produced results)
    has_data = bool(sub)

    return ChannelResult(
        'stylometry', score, severity, explanation,
        mode_eligibility=['generic_aigt'],
        sub_signals=sub,
        data_sufficient=has_data,
    )
