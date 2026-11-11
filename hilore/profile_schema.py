import math


def finite(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def shortlist(units, limit, headroom):
    if type(limit) is not int or limit < 1 or not finite(headroom):
        raise ValueError('Invalid candidate limit or memory headroom')
    ranked = []
    for index, unit in enumerate(units):
        costs = [v for k, v in unit['capture_bytes_estimate'].items()
                 if k in ('H', 'L') and finite(v, True) and v <= headroom]
        seconds = unit['recomputation_seconds_estimate']
        if costs and finite(seconds, True):
            ranked.append((seconds / min(costs), index))
    return [index for _, index in sorted(ranked, key=lambda row: (-row[0], row[1]))[:limit]]


def choose_capture(ranked, h_fraction, headroom, count):
    if h_fraction not in (0.0, 0.5, 1.0) or not finite(headroom):
        raise ValueError('Invalid capture allocation')
    q = ['R'] * count
    h_left, l_left = headroom * h_fraction, headroom * (1 - h_fraction)
    for row in ranked:
        index = row['unit']
        if row['H'] is not None and row['H']['capture_cost'] <= h_left:
            q[index] = 'H'
            h_left -= row['H']['capture_cost']
        elif row['L'] is not None and row['L']['capture_cost'] <= l_left:
            q[index] = 'L'
            l_left -= row['L']['capture_cost']
    return q


def development_acceptable(rows, budget, tolerance=0.015):
    return (bool(rows) and finite(budget, True) and finite(tolerance)
            and all(row.get('measurement_scope') == 'complete_update'
                    and finite(row.get('seconds'), True) and finite(row.get('gc_seconds'), True)
                    and finite(row.get('peak_bytes'), True) and finite(row.get('gradient_error'))
                    for row in rows)
            and max(row['peak_bytes'] for row in rows) <= budget
            and sum(row['gradient_error'] for row in rows) / len(rows) <= tolerance)


def validate_profile(profile, fingerprint, memory_multiplier, risk_budget,
                     allow_development=False, initializing=False):
    if profile.get('schema_version') != 3:
        raise ValueError('Regenerate the profile with the bounded microbatch initialization entry')
    if profile.get('configuration_fingerprint') != fingerprint:
        raise ValueError('Profile does not match the model and execution configuration')
    if not finite(risk_budget) or profile.get('risk_budget') != risk_budget:
        raise ValueError('Risk budget must match the profile')
    gc_peak = profile.get('measured_gc_peak_bytes')
    if (not finite(gc_peak, True) or not finite(memory_multiplier, True)
            or profile.get('memory_multiplier') != memory_multiplier
            or profile.get('memory_metric') != 'nvml_device_used_max'
            or profile.get('measurement_world_size') != 4):
        raise ValueError('Profile memory measurements do not match the execution configuration')
    budget = gc_peak * memory_multiplier
    if not finite(budget, True):
        raise ValueError('Invalid memory ceiling')
    protocol = profile.get('initialization', {})
    if (protocol.get('protocol') != 'bounded_microbatch_v1'
            or type(protocol.get('candidate_limit')) is not int or protocol['candidate_limit'] < 1
            or protocol.get('calibration_microbatches') != 16
            or type(protocol.get('microbatch_repeats')) is not int or protocol['microbatch_repeats'] < 2
            or protocol.get('microbatch_warmups') != 1
            or type(protocol.get('interaction_round_limit')) is not int or protocol['interaction_round_limit'] < 1
            or protocol.get('complete_update_pairs') != 1 or protocol.get('complete_update_warmups') != 0):
        raise ValueError('Invalid initialization measurement protocol')
    if initializing:
        if profile.get('status') != 'initializing' or profile.get('joint_validation_passed') is not False:
            raise ValueError('Initialization requires an explicitly unvalidated profile')
    else:
        if profile.get('status') not in ('development_only', 'validated') or profile.get('joint_validation_passed') is not True:
            raise ValueError('HiLoRe requires successful complete-update validation')
        if not allow_development and profile.get('terminal_quality_passed') is not True:
            raise ValueError('Terminal quality selection is incomplete; use --development-profile for development runs')
        rows = profile.get('development_measurements', [])
        if len(rows) != 1 or not development_acceptable(rows, budget):
            raise ValueError('Invalid complete-update validation measurements')
        counts = rows[0].get('recovery_action_counts', {})
        if (any(type(counts.get(action)) is not int or counts[action] < 0 for action in 'RHL')
                or counts['H'] + counts['L'] == 0):
            raise ValueError('Complete-update validation must execute non-R recovery')
        if rows[0]['seconds'] >= rows[0]['gc_seconds']:
            raise ValueError('The validated plan does not improve complete-update throughput')
    plans = profile.get('plans')
    if not isinstance(plans, list) or len(plans) != 1:
        raise ValueError('A bounded initialization profile must contain one selected shape plan')
    calibration = profile.get('calibration', {})
    for plan in plans:
        representations = plan['representations']
        count = len(representations)
        if (not count or len(plan['units']) != count or any(q not in ('H', 'L', 'R') for q in representations)
                or sum(q != 'R' for q in representations) > protocol['candidate_limit']
                or plan.get('measurement_scope') != 'microbatch'
                or plan.get('shape') != profile.get('calibration_shape')):
            raise ValueError('Invalid candidate plan topology or measurement scope')
        for key in ('utility_seconds', 'marginal_peak_bytes'):
            table = plan[key]
            if len(table) != count or any(len(row) != 3 or row[0] != 0 or any(not finite(v) for v in row) for row in table):
                raise ValueError('Invalid action measurements')
        for key in ('estimated_same_plan_all_r_peak_bytes', 'microbatch_all_r_peak_bytes',
                    'microbatch_all_r_seconds', 'estimated_microbatch_seconds'):
            if not finite(plan.get(key), True):
                raise ValueError('Invalid microbatch estimate')
        if plan['estimated_same_plan_all_r_peak_bytes'] > budget:
            raise ValueError('Estimated capture memory exceeds the ceiling')
        if not initializing:
            if (plan.get('interaction_screening_passed') is not True
                    or plan.get('validation_scope') != 'complete_update'
                    or not finite(plan.get('validated_peak_bytes'), True)
                    or not finite(plan.get('validated_actor_update_seconds'), True)
                    or plan['validated_peak_bytes'] > budget
                    or plan['validated_peak_bytes'] != rows[0]['peak_bytes']
                    or plan['validated_actor_update_seconds'] != rows[0]['seconds']):
                raise ValueError('The selected plan lacks complete-update validation')
        for index, unit in enumerate(plan['units']):
            actions = set(unit['actions'])
            if 'R' not in actions or not actions <= set({'R': 'R', 'H': 'RHL', 'L': 'RL'}[representations[index]]):
                raise ValueError('Recovery actions are incompatible with capture')
            if actions != {'R'} and not unit.get('recomputation_verified'):
                raise ValueError('Unverified recovery action')
            if 'L' in actions:
                calibrated = calibration.get(str(index), {})
                if (calibrated.get('group') != f'mlp/{index}'
                        or calibrated.get('selected_index') != unit.get('selected_index')
                        or not finite(calibrated.get('kappa')) or not finite(calibrated.get('mean_error'))):
                    raise ValueError('Every L action requires matching calibration')
    return budget
