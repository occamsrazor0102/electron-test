"""Reporting modules for batch analysis output.

Provides attempter profiling, channel pattern analysis, and financial impact
estimation for BEET pipeline batch results.
"""

from collections import defaultdict, Counter


def profile_attempters(results, min_submissions=2):
    """Aggregate detection results by attempter.

    Returns list of attempter profiles sorted by flag rate (descending).
    """
    by_attempter = defaultdict(list)
    for r in results:
        att = r.get('attempter', '').strip()
        if att:
            by_attempter[att].append(r)

    profiles = []
    for att, submissions in by_attempter.items():
        if len(submissions) < min_submissions:
            continue

        det_counts = Counter(r['determination'] for r in submissions)
        n_total = len(submissions)
        n_flagged = det_counts.get('RED', 0) + det_counts.get('AMBER', 0) + det_counts.get('MIXED', 0)
        flag_rate = n_flagged / n_total

        # Identify primary detection pattern for flagged submissions
        flagged_channels = Counter()
        for r in submissions:
            if r['determination'] in ('RED', 'AMBER', 'MIXED'):
                cd = r.get('channel_details', {}).get('channels', {})
                for ch_name, ch_info in cd.items():
                    if ch_info.get('severity') in ('RED', 'AMBER'):
                        flagged_channels[ch_name] += 1

        mc = flagged_channels.most_common(1) if flagged_channels else []
        primary_channel = mc[0][0] if mc else None

        # Mean confidence across flagged submissions
        flagged_confs = [r['confidence'] for r in submissions
                        if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        mean_conf = sum(flagged_confs) / len(flagged_confs) if flagged_confs else 0.0

        profiles.append({
            'attempter': att,
            'total_submissions': n_total,
            'flagged': n_flagged,
            'flag_rate': round(flag_rate, 3),
            'red': det_counts.get('RED', 0),
            'amber': det_counts.get('AMBER', 0),
            'yellow': det_counts.get('YELLOW', 0),
            'green': det_counts.get('GREEN', 0),
            'mixed': det_counts.get('MIXED', 0),
            'mean_flagged_confidence': round(mean_conf, 3),
            'primary_detection_channel': primary_channel,
            'occupations': list(set(r.get('occupation', '') for r in submissions if r.get('occupation'))),
        })

    profiles.sort(key=lambda p: (-p['flag_rate'], -p['flagged']))
    return profiles


def print_attempter_report(profiles):
    """Print attempter profiling summary."""
    if not profiles:
        print("\n  No attempter data available for profiling.")
        return

    flagged_profiles = [p for p in profiles if p['flagged'] > 0]
    clean_profiles = [p for p in profiles if p['flagged'] == 0]

    total_submissions = sum(p['total_submissions'] for p in profiles)
    total_flagged = sum(p['flagged'] for p in profiles)
    total_attempters = len(profiles)

    print(f"\n{'='*90}")
    print(f"  ATTEMPTER PROFILING: {total_attempters} contributors, "
          f"{total_submissions} submissions")
    print(f"{'='*90}")

    if flagged_profiles:
        # Concentration metric
        if total_flagged > 0:
            top_n = max(1, int(len(profiles) * 0.10))
            top_flagged = sum(p['flagged'] for p in flagged_profiles[:top_n])
            concentration = top_flagged / total_flagged * 100
            print(f"\n  Concentration: Top {top_n} contributor(s) account for "
                  f"{concentration:.0f}% of all flagged submissions")

        print(f"\n  {'Attempter':<25} {'Subs':>5} {'Flag':>5} {'Rate':>7} "
              f"{'R':>3} {'A':>3} {'Y':>3} {'G':>3} {'Primary Channel':<20}")
        print(f"  {'-'*85}")

        for p in flagged_profiles:
            ch = p['primary_detection_channel'] or '-'
            print(f"  {p['attempter'][:24]:<25} {p['total_submissions']:>5} "
                  f"{p['flagged']:>5} {p['flag_rate']:>6.0%} "
                  f"{p['red']:>3} {p['amber']:>3} {p['yellow']:>3} {p['green']:>3} "
                  f"{ch:<20}")

    if clean_profiles:
        print(f"\n  Clean contributors ({len(clean_profiles)}): "
              f"{', '.join(p['attempter'][:20] for p in clean_profiles[:10])}"
              f"{'...' if len(clean_profiles) > 10 else ''}")


def channel_pattern_summary(results):
    """Summarize flagged submissions by primary detection channel."""
    channel_counts = Counter()
    channel_examples = defaultdict(list)

    for r in results:
        if r['determination'] not in ('RED', 'AMBER', 'MIXED'):
            continue
        cd = r.get('channel_details', {}).get('channels', {})
        candidates = [
            (ch, info) for ch, info in cd.items()
            if info.get('severity') not in ('GREEN', None)
        ]
        if not candidates:
            continue
        primary = max(
            candidates,
            key=lambda x: {'RED': 3, 'AMBER': 2, 'YELLOW': 1}.get(x[1].get('severity', 'GREEN'), 0),
        )
        ch_name = primary[0]
        channel_counts[ch_name] += 1
        if len(channel_examples[ch_name]) < 3:
            channel_examples[ch_name].append(r.get('task_id', '?')[:15])

    if not channel_counts:
        return

    total = sum(channel_counts.values())
    print(f"\n  DETECTION PATTERN BREAKDOWN ({total} flagged submissions):")
    print(f"  {'-'*60}")
    for ch, count in channel_counts.most_common():
        pct = count / total * 100
        examples = ', '.join(channel_examples[ch])
        print(f"    {ch:<20} {count:>4} ({pct:>5.1f}%)  e.g. {examples}")


def financial_impact(results, cost_per_prompt=400.0):
    """Calculate financial impact of detection.

    Args:
        results: List of pipeline result dicts from batch run.
        cost_per_prompt: Dollar cost per commissioned prompt.

    Returns dict with impact metrics.
    """
    n_total = len(results)
    if n_total == 0:
        return {
            'total_submissions': 0,
            'total_spend': 0.0,
            'flagged_count': 0,
            'flag_rate': 0.0,
            'waste_estimate': 0.0,
            'clean_yield': 0.0,
            'clean_count': 0,
            'projected_annual_waste': 0.0,
            'projected_annual_savings_60pct': 0.0,
        }

    det_counts = Counter(r['determination'] for r in results)

    n_flagged = det_counts.get('RED', 0) + det_counts.get('AMBER', 0) + det_counts.get('MIXED', 0)

    flag_rate = n_flagged / n_total
    total_spend = n_total * cost_per_prompt
    waste_at_flag = n_flagged * cost_per_prompt

    clean_count = n_total - n_flagged
    clean_yield = clean_count / n_total

    annual_waste = waste_at_flag * 4
    annual_savings_at_60pct = annual_waste * 0.60

    return {
        'total_submissions': n_total,
        'total_spend': total_spend,
        'flagged_count': n_flagged,
        'flag_rate': round(flag_rate, 3),
        'waste_estimate': waste_at_flag,
        'clean_yield': round(clean_yield, 3),
        'clean_count': clean_count,
        'projected_annual_waste': annual_waste,
        'projected_annual_savings_60pct': annual_savings_at_60pct,
    }


def print_financial_report(impact, cost_per_prompt=400.0):
    """Print financial impact summary."""
    print(f"\n{'='*90}")
    print(f"  FINANCIAL IMPACT ESTIMATE (${cost_per_prompt:.0f}/prompt)")
    print(f"{'='*90}")
    print(f"    Total submissions:       {impact['total_submissions']:>8}")
    print(f"    Total spend:             ${impact['total_spend']:>10,.0f}")
    print(f"    Flagged (RED+AMBER):     {impact['flagged_count']:>8}  "
          f"({impact['flag_rate']:.1%})")
    print(f"    Estimated waste:         ${impact['waste_estimate']:>10,.0f}")
    print(f"    Clean yield:             {impact['clean_count']:>8}  "
          f"({impact['clean_yield']:.1%})")
    print(f"")
    print(f"    Projected annual waste:  ${impact['projected_annual_waste']:>10,.0f}  "
          f"(4 quarterly batches)")
    print(f"    Annual savings (60%):    ${impact['projected_annual_savings_60pct']:>10,.0f}  "
          f"(conservative catch rate)")
