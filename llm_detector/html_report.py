"""HTML report generator with span-level highlighting."""

import html


_CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 900px;
       margin: 40px auto; padding: 0 20px; background: #fafafa; color: #1a1a1a; }
.header { border-bottom: 3px solid #1a1a1a; padding-bottom: 16px; margin-bottom: 24px; }
.det { font-size: 28px; font-weight: 700; }
.det-RED { color: #d32f2f; }
.det-AMBER { color: #f57c00; }
.det-YELLOW { color: #fbc02d; }
.det-GREEN { color: #388e3c; }
.det-MIXED { color: #1976d2; }
.meta { color: #666; font-size: 14px; margin-top: 8px; }
.text-container { background: white; border: 1px solid #e0e0e0; border-radius: 8px;
                  padding: 24px; line-height: 1.8; font-size: 15px; white-space: pre-wrap;
                  word-wrap: break-word; }
.signal { padding: 2px 0; border-bottom: 3px solid; cursor: help; }
.signal-CRITICAL { border-color: #ff1744; background: #ffebee; }
.signal-HIGH { border-color: #ff5722; background: #fbe9e7; }
.signal-MEDIUM { border-color: #ff9800; background: #fff3e0; }
.signal-pattern { border-color: #ff9800; background: #fff8e1; }
.signal-keyword { border-color: #42a5f5; background: #e3f2fd; }
.signal-uppercase { border-color: #e53935; background: #ffcdd2; }
.signal-fingerprint { border-color: #ab47bc; background: #f3e5f5; }
.signal-hot_window { border-color: #ef5350; background: #ffcdd2; }
.legend { margin-top: 24px; padding: 16px; background: #f5f5f5; border-radius: 8px;
          font-size: 13px; }
.legend span { display: inline-block; margin-right: 16px; padding: 2px 6px; }
.channels { margin-top: 24px; }
.ch-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.ch-table th { text-align: left; padding: 8px 10px; background: #e8e8e8;
               font-weight: 600; font-size: 13px; }
.ch-table td { padding: 8px 10px; border-bottom: 1px solid #eee; vertical-align: middle; }
.ch-name { font-weight: 600; white-space: nowrap; }
.ch-score-bar { width: 90px; }
.score-bar-bg { background: #e0e0e0; border-radius: 4px; height: 10px; width: 80px; }
.score-bar-fill { height: 10px; border-radius: 4px; }
.score-bar-RED { background: #d32f2f; }
.score-bar-AMBER { background: #f57c00; }
.score-bar-YELLOW { background: #fbc02d; }
.score-bar-GREEN { background: #388e3c; }
.score-val { font-size: 12px; color: #555; margin-top: 2px; }
.role-badge { display: inline-block; font-size: 11px; font-weight: 600;
              padding: 2px 7px; border-radius: 10px; white-space: nowrap; }
.role-primary { background: #e3f2fd; color: #1565c0; }
.role-supporting { background: #f3e5f5; color: #6a1b9a; }
.role-nodata { background: #f5f5f5; color: #757575; }
.role-disabled { background: #fbe9e7; color: #b71c1c; }
.ch-expl { font-size: 13px; color: #444; }
.fusion-box { margin-top: 24px; padding: 16px; background: #fffde7;
              border: 1px solid #ffe082; border-radius: 8px; font-size: 13px; }
.fusion-box h3 { margin: 0 0 10px 0; font-size: 15px; }
.fusion-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
.fusion-item { background: white; border: 1px solid #ffe082; border-radius: 6px;
               padding: 6px 12px; font-size: 13px; }
.fusion-label { color: #666; font-size: 12px; }
.fusion-val { font-weight: 600; font-size: 14px; }
.rule-badge { display: inline-block; background: #fff9c4; border: 1px solid #f9a825;
              border-radius: 4px; padding: 3px 10px; font-size: 12px;
              font-weight: 600; color: #e65100; }
"""


_CHANNEL_LABELS = {
    'prompt_structure': 'Prompt Structure',
    'stylometry': 'Stylometry',
    'continuation': 'Continuation (DNA-GPT)',
    'windowing': 'Windowing',
}

_TRIGGERING_RULE_LABELS = {
    'L0_critical_preamble': 'L0 Critical Preamble — instant RED',
    'primary_red_with_corroboration': 'Primary RED channel + ≥2 YELLOW+ channels → RED',
    'primary_red_short_text_relaxed': 'Primary RED + 1 YELLOW (short-text relaxation) → RED',
    'two_primary_amber_channels': '2 Primary AMBER channels → RED',
    'primary_red_single_channel_demoted': 'Single primary RED (no corroboration) → AMBER',
    'generic_aigt_red_with_corroboration': 'Generic-AIGT RED channel + ≥2 YELLOW+ → RED',
    'generic_aigt_red_single_channel': 'Generic-AIGT single RED channel → RED (capped at 75%)',
    'primary_amber_single_channel': 'Single primary AMBER channel → AMBER',
    'primary_amber_mixed_windowing': 'Primary AMBER + windowing mixed-signal → MIXED',
    'multi_channel_convergence': 'Multi-channel convergence (≥2 YELLOW+) → AMBER',
    'multi_channel_convergence_mixed': 'Multi-channel convergence + windowing mixed-signal → MIXED',
    'supporting_channel_amber': 'Supporting channel at AMBER → AMBER',
    'yellow_signal': 'Single YELLOW+ channel → YELLOW',
    'yellow_signal_mixed_windowing': 'YELLOW + windowing mixed-signal → MIXED',
    'obfuscation_delta': 'Obfuscation normalization delta → YELLOW',
    'weak_signals_below_threshold': 'Weak signals below threshold → REVIEW',
    'no_signal': 'No significant signals → GREEN',
}


def _build_channel_rows(cd):
    """Build HTML table rows for channel scores, including score bars and role badges."""
    rows = []
    channel_order = ['prompt_structure', 'stylometry', 'continuation', 'windowing']
    # Include any channels present in cd but not in the standard order
    additional_channels = [k for k in cd if k not in channel_order]
    for ch_name in channel_order + additional_channels:
        info = cd.get(ch_name, {})
        if not info:
            continue
        sev = info.get('severity', 'GREEN')
        score = info.get('score', 0.0)
        expl = info.get('explanation', '')
        role = info.get('role', '')
        data_sufficient = info.get('data_sufficient', True)
        disabled = info.get('disabled', False)
        label = html.escape(_CHANNEL_LABELS.get(ch_name, ch_name))

        # Score bar
        bar_width = int(score * 80)
        score_bar = (
            f'<div class="score-bar-bg">'
            f'<div class="score-bar-fill score-bar-{sev}" style="width:{bar_width}px"></div>'
            f'</div>'
            f'<div class="score-val">{score:.2f}</div>'
        )

        # Role badge
        if disabled:
            role_html = '<span class="role-badge role-disabled">Disabled</span>'
        elif not data_sufficient:
            role_html = '<span class="role-badge role-nodata">No Data</span>'
        elif role == 'supporting':
            role_html = '<span class="role-badge role-supporting">Supporting</span>'
        else:
            role_html = '<span class="role-badge role-primary">Primary</span>'

        rows.append(
            f'<tr>'
            f'<td class="ch-name">{label}</td>'
            f'<td class="ch-score-bar">{score_bar}</td>'
            f'<td class="det-{sev}" style="font-weight:600">{html.escape(sev)}</td>'
            f'<td>{role_html}</td>'
            f'<td class="ch-expl">{html.escape(expl)}</td>'
            f'</tr>'
        )
    return '\n'.join(rows)


def _build_fusion_section(channel_details, heading_level=3):
    """Build the Determination Basis HTML section from channel_details.

    Args:
        channel_details: The channel_details dict from fusion output.
        heading_level: HTML heading level to use (default 3 for h3; use 4 for batch sub-sections).
    """
    if not channel_details:
        return ''

    hl = heading_level
    mode = channel_details.get('mode', '?')
    fc = channel_details.get('fusion_counts', {})
    rule = channel_details.get('triggering_rule', '')
    rule_label = html.escape(_TRIGGERING_RULE_LABELS.get(rule, rule or 'Unknown'))

    items = [
        ('Mode', html.escape(mode)),
        ('Primary RED channels', str(fc.get('n_primary_red', '—'))),
        ('Primary AMBER+ channels', str(fc.get('n_primary_amber', '—'))),
        ('All YELLOW+ channels', str(fc.get('n_yellow_plus', '—'))),
        ('All RED channels', str(fc.get('n_red', '—'))),
    ]
    items_html = ''.join(
        f'<div class="fusion-item">'
        f'<div class="fusion-label">{k}</div>'
        f'<div class="fusion-val">{v}</div>'
        f'</div>'
        for k, v in items
    )

    disabled = channel_details.get('disabled_channels', [])
    disabled_note = (
        f'<div style="margin-top:10px;font-size:12px;color:#888">'
        f'Ablated channels: {html.escape(", ".join(disabled))}</div>'
        if disabled else ''
    )

    short_text_note = (
        '<div style="margin-top:6px;font-size:12px;color:#e65100">'
        '⚠ Short-text adjustment applied (word count &lt; 100)</div>'
        if channel_details.get('short_text_adjustment') else ''
    )

    return (
        f'<div class="fusion-box">'
        f'<h{hl}>Determination Basis</h{hl}>'
        f'<div style="margin-bottom:10px">'
        f'Triggering rule: <span class="rule-badge">{rule_label}</span>'
        f'</div>'
        f'<div class="fusion-row">{items_html}</div>'
        f'{short_text_note}{disabled_note}'
        f'</div>'
    )


def generate_html_report(text, result, output_path=None):
    """Generate an HTML report with highlighted detection spans.

    Args:
        text: Original input text.
        result: Pipeline result dict (must include 'detection_spans').
        output_path: Where to write the HTML file. If None, returns string.

    Returns HTML string, or writes to file and returns path.
    """
    spans = result.get('detection_spans', [])
    det = result.get('determination', 'GREEN')
    reason = result.get('reason', '')
    confidence = result.get('confidence', 0)
    task_id = result.get('task_id', '')
    word_count = result.get('word_count', 0)

    # Filter to character-level spans and sort
    char_spans = sorted(
        [s for s in spans if 'start' in s and 'end' in s],
        key=lambda s: s['start'],
    )

    highlighted = _apply_highlights(text, char_spans)

    # Channel summary
    cd = result.get('channel_details', {}).get('channels', {})
    channel_rows = _build_channel_rows(cd)

    # Determination basis section
    fusion_section = _build_fusion_section(result.get('channel_details', {}))

    report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BEET Detection Report</title>
<style>{_CSS}</style></head><body>
<div class="header">
    <div class="det det-{det}">{det}</div>
    <div class="meta">
        Task: {html.escape(task_id)} | Words: {word_count} |
        Confidence: {confidence:.1%} |
        Mode: {result.get('mode', '?')}
    </div>
    <div class="meta" style="margin-top:4px">{html.escape(reason[:200])}</div>
</div>
<div class="text-container">{highlighted}</div>
<div class="legend">
    <strong>Legend:</strong>
    <span class="signal signal-CRITICAL">CRITICAL preamble</span>
    <span class="signal signal-HIGH">HIGH preamble</span>
    <span class="signal signal-pattern">Lexicon pack hit</span>
    <span class="signal signal-keyword">Keyword match</span>
    <span class="signal signal-uppercase">Uppercase normative</span>
    <span class="signal signal-fingerprint">Fingerprint word</span>
    <span class="signal signal-hot_window">Hot window</span>
</div>
<div class="channels">
    <h3>Channel Scores</h3>
    <table class="ch-table">
        <tr>
            <th>Channel</th>
            <th>Score</th>
            <th>Severity</th>
            <th>Role</th>
            <th>Detail</th>
        </tr>
        {channel_rows}
    </table>
</div>
{fusion_section}
</body></html>"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return output_path

    return report


_BATCH_CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 960px;
       margin: 40px auto; padding: 0 20px; background: #fafafa; color: #1a1a1a; }
.batch-header { border-bottom: 3px solid #1a1a1a; padding-bottom: 16px; margin-bottom: 24px; }
.batch-header h1 { margin: 0 0 8px 0; }
.summary-table { width: 100%; border-collapse: collapse; margin-bottom: 32px; font-size: 14px; }
.summary-table th { text-align: left; padding: 8px 12px; background: #e0e0e0; }
.summary-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.submission { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 32px;
              background: white; overflow: hidden; }
.sub-header { padding: 16px 24px; border-bottom: 1px solid #e0e0e0; }
.det { font-weight: 700; }
.det-RED { color: #d32f2f; }
.det-AMBER { color: #f57c00; }
.det-YELLOW { color: #fbc02d; }
.det-GREEN { color: #388e3c; }
.det-MIXED { color: #1976d2; }
.meta { color: #666; font-size: 14px; margin-top: 4px; }
.text-container { padding: 24px; line-height: 1.8; font-size: 15px; white-space: pre-wrap;
                  word-wrap: break-word; }
.signal { padding: 2px 0; border-bottom: 3px solid; cursor: help; }
.signal-CRITICAL { border-color: #ff1744; background: #ffebee; }
.signal-HIGH { border-color: #ff5722; background: #fbe9e7; }
.signal-MEDIUM { border-color: #ff9800; background: #fff3e0; }
.signal-pattern { border-color: #ff9800; background: #fff8e1; }
.signal-keyword { border-color: #42a5f5; background: #e3f2fd; }
.signal-uppercase { border-color: #e53935; background: #ffcdd2; }
.signal-fingerprint { border-color: #ab47bc; background: #f3e5f5; }
.signal-hot_window { border-color: #ef5350; background: #ffcdd2; }
.channels { padding: 0 24px 12px; }
.ch-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ch-table th { text-align: left; padding: 6px 8px; background: #e8e8e8; font-weight: 600; font-size: 12px; }
.ch-table td { padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: middle; }
.ch-name { font-weight: 600; white-space: nowrap; }
.ch-score-bar { width: 80px; }
.score-bar-bg { background: #e0e0e0; border-radius: 4px; height: 8px; width: 70px; }
.score-bar-fill { height: 8px; border-radius: 4px; }
.score-bar-RED { background: #d32f2f; }
.score-bar-AMBER { background: #f57c00; }
.score-bar-YELLOW { background: #fbc02d; }
.score-bar-GREEN { background: #388e3c; }
.score-val { font-size: 11px; color: #555; margin-top: 2px; }
.role-badge { display: inline-block; font-size: 11px; font-weight: 600;
              padding: 1px 6px; border-radius: 10px; white-space: nowrap; }
.role-primary { background: #e3f2fd; color: #1565c0; }
.role-supporting { background: #f3e5f5; color: #6a1b9a; }
.role-nodata { background: #f5f5f5; color: #757575; }
.role-disabled { background: #fbe9e7; color: #b71c1c; }
.ch-expl { font-size: 12px; color: #444; }
.fusion-box { margin: 0 24px 16px; padding: 12px; background: #fffde7;
              border: 1px solid #ffe082; border-radius: 8px; font-size: 12px; }
.fusion-box h4 { margin: 0 0 8px 0; font-size: 13px; }
.fusion-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.fusion-item { background: white; border: 1px solid #ffe082; border-radius: 6px;
               padding: 4px 10px; font-size: 12px; }
.fusion-label { color: #666; font-size: 11px; }
.fusion-val { font-weight: 600; font-size: 13px; }
.rule-badge { display: inline-block; background: #fff9c4; border: 1px solid #f9a825;
              border-radius: 4px; padding: 2px 8px; font-size: 11px;
              font-weight: 600; color: #e65100; }
.legend { margin: 24px 0; padding: 16px; background: #f5f5f5; border-radius: 8px;
          font-size: 13px; }
.legend span { display: inline-block; margin-right: 16px; padding: 2px 6px; }
.toc { margin-bottom: 32px; }
.toc a { color: #1976d2; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""


def generate_batch_html_report(results, text_map, output_path=None):
    """Generate a single consolidated HTML report for multiple submissions.

    Args:
        results: List of pipeline result dicts (typically flagged ones).
        text_map: Dict mapping task_id -> original text.
        output_path: Where to write the HTML file. If None, returns string.

    Returns HTML string, or writes to file and returns path.
    """
    from datetime import datetime

    n_red = sum(1 for r in results if r.get('determination') == 'RED')
    n_amber = sum(1 for r in results if r.get('determination') == 'AMBER')
    n_mixed = sum(1 for r in results if r.get('determination') == 'MIXED')
    n_yellow = sum(1 for r in results if r.get('determination') == 'YELLOW')

    # Table of contents
    toc_rows = []
    for idx, r in enumerate(results):
        tid = r.get('task_id', f'submission_{idx}')
        det = r.get('determination', 'GREEN')
        conf = r.get('confidence', 0)
        att = r.get('attempter', '')
        label = html.escape(att or tid)
        toc_rows.append(
            f'<tr><td><a href="#sub-{idx}">{label}</a></td>'
            f'<td class="det det-{det}">{det}</td>'
            f'<td>{conf:.1%}</td>'
            f'<td>{html.escape(tid)}</td></tr>'
        )

    # Submission sections
    sections = []
    for idx, r in enumerate(results):
        tid = r.get('task_id', f'submission_{idx}')
        det = r.get('determination', 'GREEN')
        reason = r.get('reason', '')
        confidence = r.get('confidence', 0)
        word_count = r.get('word_count', 0)
        att = r.get('attempter', '')
        text = text_map.get(tid, '')

        spans = r.get('detection_spans', [])
        char_spans = sorted(
            [s for s in spans if 'start' in s and 'end' in s],
            key=lambda s: s['start'],
        )
        highlighted = _apply_highlights(text, char_spans)

        cd = r.get('channel_details', {}).get('channels', {})
        channel_rows = _build_channel_rows(cd)
        ch_table = (
            f'<table class="ch-table">'
            f'<tr><th>Channel</th><th>Score</th><th>Severity</th><th>Role</th><th>Detail</th></tr>'
            f'{channel_rows}'
            f'</table>'
        )
        fusion_section = _build_fusion_section(r.get('channel_details', {}), heading_level=4)

        att_label = f' | Fellow: {html.escape(att)}' if att else ''
        sections.append(f"""
<div class="submission" id="sub-{idx}">
    <div class="sub-header">
        <span class="det det-{det}" style="font-size:22px">{det}</span>
        <span class="meta" style="margin-left:12px">
            Task: {html.escape(tid)}{att_label} | Words: {word_count} |
            Confidence: {confidence:.1%} | Mode: {r.get('mode', '?')}
        </span>
        <div class="meta" style="margin-top:4px">{html.escape(reason[:200])}</div>
    </div>
    <div class="text-container">{highlighted}</div>
    <div class="channels">{ch_table}</div>
    {fusion_section}
</div>""")

    report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BEET Batch Detection Report</title>
<style>{_BATCH_CSS}</style></head><body>
<div class="batch-header">
    <h1>BEET Batch Detection Report</h1>
    <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
        Submissions: {len(results)} |
        <span class="det-RED">RED: {n_red}</span> |
        <span class="det-AMBER">AMBER: {n_amber}</span> |
        <span class="det-MIXED">MIXED: {n_mixed}</span> |
        <span class="det-YELLOW">YELLOW: {n_yellow}</span>
    </div>
</div>
<div class="toc">
    <h3>Flagged Submissions</h3>
    <table class="summary-table">
        <tr><th>Fellow / ID</th><th>Determination</th><th>Confidence</th><th>Task ID</th></tr>
        {''.join(toc_rows)}
    </table>
</div>
<div class="legend">
    <strong>Legend:</strong>
    <span class="signal signal-CRITICAL">CRITICAL preamble</span>
    <span class="signal signal-HIGH">HIGH preamble</span>
    <span class="signal signal-pattern">Lexicon pack hit</span>
    <span class="signal signal-keyword">Keyword match</span>
    <span class="signal signal-uppercase">Uppercase normative</span>
    <span class="signal signal-fingerprint">Fingerprint word</span>
    <span class="signal signal-hot_window">Hot window</span>
</div>
{''.join(sections)}
</body></html>"""

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return output_path

    return report


def _get_span_class(span):
    """Determine CSS class for a span based on its type/severity."""
    sev = span.get('severity')
    if sev in ('CRITICAL', 'HIGH', 'MEDIUM'):
        return sev
    span_type = span.get('type', 'pattern')
    if span_type in ('fingerprint', 'hot_window', 'keyword', 'uppercase'):
        return span_type
    return 'pattern'


_SEVERITY_ORDER = {
    'CRITICAL': 6, 'HIGH': 5, 'MEDIUM': 4,
    'uppercase': 3, 'fingerprint': 2, 'pattern': 1, 'keyword': 0,
    'hot_window': 1,
}


def _apply_highlights(text, spans):
    """Apply highlight markup to text at span positions.

    Handles overlapping spans by using the highest-severity span at each position.
    """
    if not spans:
        return html.escape(text)

    # Build a priority map: each character position gets the highest-severity annotation
    char_map = [None] * len(text)
    for span in spans:
        start = span.get('start', 0)
        end = span.get('end', start)
        css_class = _get_span_class(span)
        tooltip = span.get('pack', span.get('pattern', span.get('source', '')))
        priority = _SEVERITY_ORDER.get(css_class, 0)
        for i in range(max(0, start), min(end, len(text))):
            if char_map[i] is None or priority > char_map[i][2]:
                char_map[i] = (css_class, tooltip, priority)

    # Build output with runs of same annotation
    out = []
    i = 0
    while i < len(text):
        if char_map[i] is None:
            j = i
            while j < len(text) and char_map[j] is None:
                j += 1
            out.append(html.escape(text[i:j]))
            i = j
        else:
            css_class, tooltip, _ = char_map[i]
            j = i
            while j < len(text) and char_map[j] is not None and char_map[j][0] == css_class and char_map[j][1] == tooltip:
                j += 1
            out.append(
                f'<span class="signal signal-{css_class}" title="{html.escape(tooltip)}">'
                f'{html.escape(text[i:j])}</span>'
            )
            i = j

    return ''.join(out)
