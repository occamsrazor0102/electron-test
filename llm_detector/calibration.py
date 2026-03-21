"""Conformal calibration from labeled baseline data.

Ref: Vovk et al. (2005), Shafer & Vovk (2008)
"""

import json
import math
from collections import defaultdict

_CALIBRATION_ALPHAS = [0.01, 0.05, 0.10]


def calibrate_from_baselines(jsonl_path):
    """Build calibration tables from labeled baseline data."""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if rec.get('ground_truth') == 'human':
                        records.append(rec)
                except json.JSONDecodeError:
                    continue

    if len(records) < 20:
        return None

    nc_scores = [1.0 - float(r.get('confidence', 0)) for r in records]
    nc_scores.sort()

    global_cal = {}
    for alpha in _CALIBRATION_ALPHAS:
        idx = int(math.ceil((1 - alpha) * (len(nc_scores) + 1))) - 1
        idx = max(0, min(idx, len(nc_scores) - 1))
        global_cal[alpha] = nc_scores[idx]

    strata = defaultdict(list)
    for r in records:
        domain = r.get('domain', 'unknown') or 'unknown'
        length_bin = r.get('length_bin', 'unknown') or 'unknown'
        conf = float(r.get('confidence', 0))
        strata[(domain, length_bin)].append(1.0 - conf)

    strata_cal = {}
    strata_counts = {}
    for key, scores in strata.items():
        scores.sort()
        strata_counts[key] = len(scores)
        if len(scores) >= 10:
            strata_cal[key] = {}
            for alpha in _CALIBRATION_ALPHAS:
                idx = int(math.ceil((1 - alpha) * (len(scores) + 1))) - 1
                idx = max(0, min(idx, len(scores) - 1))
                strata_cal[key][alpha] = scores[idx]

    return {
        'global': global_cal,
        'strata': strata_cal,
        'n_calibration': len(records),
        'strata_counts': {f"{k[0]}_{k[1]}": v for k, v in strata_counts.items()},
    }


def apply_calibration(confidence, cal_table, domain=None, length_bin=None):
    """Apply conformal calibration to a raw confidence score.

    Returns dict with:
        conformity_level: Discretized measure of how typical this confidence
            score is among calibrated human-authored texts. Values:
            1.0  = highly typical of human text (low nonconformity)
            0.10 = moderately unusual among human calibration set
            0.05 = quite unusual among human calibration set
            0.01 = very unusual — strong signal of non-human origin
    """
    if cal_table is None:
        return {
            'raw_confidence': confidence,
            'calibrated_confidence': confidence,
            'conformity_level': None,
            'stratum_used': 'uncalibrated',
        }

    nc_score = 1.0 - confidence

    stratum_key = (domain or 'unknown', length_bin or 'unknown')
    if stratum_key in cal_table.get('strata', {}):
        cal = cal_table['strata'][stratum_key]
        stratum_label = f"{stratum_key[0]}_{stratum_key[1]}"
    else:
        cal = cal_table.get('global', {})
        stratum_label = 'global'

    if nc_score <= cal.get(0.01, 0):
        conformity_level = 1.0
    elif nc_score <= cal.get(0.05, 0):
        conformity_level = 0.10
    elif nc_score <= cal.get(0.10, 0):
        conformity_level = 0.05
    else:
        conformity_level = 0.01

    alpha_05 = cal.get(0.05, 0.5)
    if nc_score > alpha_05:
        calibrated = min(confidence * 1.15, 0.99)
    elif nc_score < cal.get(0.10, 0.5):
        calibrated = confidence * 0.75
    else:
        calibrated = confidence

    return {
        'raw_confidence': round(confidence, 4),
        'calibrated_confidence': round(calibrated, 4),
        'conformity_level': round(conformity_level, 4) if conformity_level is not None else None,
        'stratum_used': stratum_label,
    }


def save_calibration(cal_table, path):
    """Save calibration table to JSON."""
    serializable = {
        'global': cal_table['global'],
        'strata': {f"{k[0]}|{k[1]}": v for k, v in cal_table.get('strata', {}).items()}
                  if isinstance(list(cal_table.get('strata', {}).keys() or [('',)])[0], tuple)
                  else cal_table.get('strata', {}),
        'n_calibration': cal_table['n_calibration'],
        'strata_counts': cal_table.get('strata_counts', {}),
    }
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Calibration table saved to {path} ({cal_table['n_calibration']} records)")


def load_calibration(path):
    """Load calibration table from JSON."""
    with open(path, 'r') as f:
        raw = json.load(f)

    strata = {}
    for k, v in raw.get('strata', {}).items():
        parts = k.split('|')
        if len(parts) == 2:
            strata[(parts[0], parts[1])] = {float(ak): av for ak, av in v.items()}
        else:
            strata[k] = v

    return {
        'global': {float(k): v for k, v in raw.get('global', {}).items()},
        'strata': strata,
        'n_calibration': raw.get('n_calibration', 0),
        'strata_counts': raw.get('strata_counts', {}),
    }
