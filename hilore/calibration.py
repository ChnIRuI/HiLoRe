"""Initialization fitting and measured schedule acceptance."""

import math
from collections import defaultdict


def fit(rows, manifest=None):
    """Fit Eq. 22 and freeze Eq. 23 from 16 isolated training microbatches.

    Each row supplies measured bundle error, isolated gradient error, current
    exposure, a stable unit/group identifier and a calibration microbatch ID.
    """
    if manifest is None:
        raise ValueError('Calibration requires the frozen problem manifest')
    if not rows:
        raise ValueError('Calibration observations are required')
    by_unit, by_group = defaultdict(list), defaultdict(list)
    for row in rows:
        manifest.require_usage(row.get('problem_ids'), 'calibration', row.get('split_manifest_sha256'))
        if row['split'] != 'train' or row['intervention_count'] != 1:
            raise ValueError("Calibration requires isolated training interventions")
        if not row['paired_state_verified']:
            raise ValueError("Calibration pairs must preserve parameters, inputs and RNG")
        x = float(row['exposure']) * float(row['reconstruction_error'])
        y = float(row['gradient_error'])
        if not all(math.isfinite(v) and v >= 0 for v in (x, y, row['exposure'], row['reconstruction_error'])):
            raise ValueError("Calibration measurements must be finite and nonnegative")
        by_unit[row['unit']].append(row)
        by_group[row['group']].append((x, y))
    kappa = {}
    for group, points in by_group.items():
        denominator = sum(x * x for x, y in points)
        if denominator == 0:
            if any(y > 0 for x, y in points):
                raise ValueError("Positive drift with zero calibration exposure; reject L group")
            kappa[group] = 0.0
        else:
            kappa[group] = max(0.0, sum(x * y for x, y in points) / denominator)
    result = {}
    microbatches = None
    for unit, samples in by_unit.items():
        if len(samples) != 16 or len({r['microbatch'] for r in samples}) != 16:
            raise ValueError("Each L unit requires 16 distinct training microbatches")
        current = {r['microbatch'] for r in samples}
        if microbatches is not None and current != microbatches:
            raise ValueError('All units must use the same 16 calibration microbatches')
        microbatches = current
        groups = {r['group'] for r in samples}
        if len(groups) != 1:
            raise ValueError("A unit must have a stable state-type/layer group")
        group = next(iter(groups))
        result[str(unit)] = {'group': group, 'kappa': kappa[group],
                             'mean_error': sum(r['reconstruction_error'] for r in samples) / 16}
    return result


def accept_updates(measurements, memory_budget, gc_validation_score, validation_score):
    """Apply Eqs. 30 and 53 to complete updates, never microbatch norms."""
    if not measurements:
        raise ValueError("Complete-update measurements are required")
    if any(not m['complete_update'] or not m['after_reduction_before_clipping'] for m in measurements):
        raise ValueError("Invalid gradient measurement boundary")
    numbers = [float(m[k]) for m in measurements for k in ('peak_bytes', 'gradient_error')]
    if not math.isfinite(memory_budget) or memory_budget <= 0:
        raise ValueError('The memory ceiling must be finite and positive')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (gc_validation_score, validation_score)):
        raise ValueError('Validation scores must be finite fractions')
    if not all(math.isfinite(v) and v >= 0 for v in numbers):
        return False
    return (max(m['peak_bytes'] for m in measurements) <= memory_budget
            and sum(m['gradient_error'] for m in measurements) / len(measurements) <= 0.015
            and validation_score >= gc_validation_score - 0.01)
