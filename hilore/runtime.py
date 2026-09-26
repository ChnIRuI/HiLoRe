"""Pre-forward candidate plans and post-forward current-update allocation."""

import json
import math
from pathlib import Path
from .allocator import allocate, warmup
from .objective import exposure


class Scheduler:
    def __init__(self, profile_path, configuration_fingerprint, memory_multiplier, risk_budget,
                 allow_development=False, action_transform=None):
        self.action_transform = action_transform
        if risk_budget is None or not math.isfinite(risk_budget) or risk_budget < 0:
            raise ValueError('Select the surrogate risk budget from development measurements')
        self.profile = profile_path if isinstance(profile_path, dict) else json.loads(Path(profile_path).read_text())
        if self.profile.get('schema_version') != 2:
            raise ValueError('Regenerate the profile with the current measured initialization entry')
        if self.profile['configuration_fingerprint'] != configuration_fingerprint:
            raise ValueError('Profile does not match model, precision, packing, software and FSDP configuration')
        if not self.profile.get('joint_validation_passed'):
            raise ValueError('HiLoRe requires complete-update schedule validation')
        if not allow_development and not self.profile.get('terminal_quality_passed'):
            raise ValueError('Terminal quality selection is incomplete; use --development-profile only for development runs')
        if (self.profile.get('memory_metric') != 'nvml_device_used_max'
                or self.profile.get('measurement_world_size') != 4
                or self.profile.get('memory_multiplier') != memory_multiplier):
            raise ValueError('Profile memory measurement does not match the paper protocol')
        gc_peak = float(self.profile['measured_gc_peak_bytes'])
        if not all(math.isfinite(v) and v > 0 for v in (gc_peak, memory_multiplier)):
            raise ValueError('GC peak and budget multiplier must be finite and positive')
        self.budget = gc_peak * memory_multiplier
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError('A matched measured GC peak is required')
        if self.profile['risk_budget'] != risk_budget:
            raise ValueError('The risk budget must match the frozen development selection')
        self.risk_budget = risk_budget
        self.plan = None
        for plan in self.profile['plans']:
            n = len(plan['representations'])
            if len(plan['units']) != n or any(q not in ('H', 'L', 'R') for q in plan['representations']):
                raise ValueError('Invalid profile topology')
            for key in ('utility_seconds', 'marginal_peak_bytes'):
                table = plan[key]
                if len(table) != n or any(len(row) != 3 or row[0] != 0 or
                    any(not math.isfinite(v) or v < 0 for v in row) for row in table):
                    raise ValueError('Invalid measured action table')
            for key in ('same_plan_all_r_peak_bytes', 'validated_peak_bytes', 'validated_actor_update_seconds'):
                if not math.isfinite(plan[key]) or plan[key] <= 0:
                    raise ValueError('Profile measurements must be finite and positive')
            for unit in plan['units']:
                if not set(unit['actions']) <= set('RHL') or 'R' not in unit['actions']:
                    raise ValueError('Invalid profile action set')
        warmup()

    def before_forward(self, shape, layer_count):
        self.shape = list(shape)
        candidates = [p for p in self.profile['plans'] if p['shape'] == list(shape)
                      and p['same_plan_all_r_peak_bytes'] <= self.budget
                      and p['validated_peak_bytes'] <= self.budget]
        self.plan = min(candidates, key=lambda p: p['validated_actor_update_seconds'], default=None)
        if self.plan is None:
            return ['R'] * layer_count
        if len(self.plan['representations']) != layer_count:
            raise ValueError('Profile unit topology does not match the learner')
        return self.plan['representations']

    def select(self, recovery, actions):
        if self.action_transform is not None:
            actions = self.action_transform(self.shape, list(actions))
        return recovery.select(actions)

    def after_forward(self, recovery, omega, response_mask, microbatch_tokens, update_tokens):
        if update_tokens <= 0 or not 0 <= microbatch_tokens <= update_tokens:
            raise ValueError('Invalid complete-update token normalization')
        if self.plan is None:
            actions = ['R'] * len(recovery.layers)
            return self.select(recovery, actions)
        plan = self.plan
        if len(recovery.bundles) != len(plan['units']):
            raise ValueError('Captured unit topology differs from the profile')
        risks, available = [], []
        current_exposure = None
        for i, unit in enumerate(plan['units']):
            bundle = recovery.bundles[i]
            signature = [[list(a), list(b), c] for a, b, c, device in bundle.records]
            if signature != unit['saved_state_signature'] or bundle.selected != unit.get('selected_index'):
                actions = ['R'] * len(recovery.layers)
                return self.select(recovery, actions)
            if not unit['recomputation_verified']:
                raise ValueError('Unverified recovery unit in profile')
            actions = set(unit['actions']) & bundle.available
            available.append(actions)
            if 'L' in actions:
                calibration = self.profile['calibration'][str(i)]
                if calibration['group'] != f'mlp/{i}':
                    raise ValueError('Unsupported susceptibility group')
                if calibration.get('selected_index') != bundle.selected:
                    raise ValueError('Calibration refers to a different compressed component')
                # A complete decoder bundle spans all dense source positions.
                # Its selected MLP component reaches all valid response positions.
                if unit['support_kind'] != 'complete_dense_decoder':
                    raise ValueError('Unrecognized structural support')
                if not all(math.isfinite(calibration[k]) and calibration[k] >= 0 for k in ('kappa', 'mean_error')):
                    raise ValueError('Invalid frozen calibration statistic')
                if current_exposure is None:
                    current_exposure = exposure(omega, response_mask.bool())
                risk = calibration['kappa'] * calibration['mean_error'] * current_exposure
            else:
                risk = 0.0
            risks.append(risk)
        actions, _ = allocate(plan['utility_seconds'], plan['marginal_peak_bytes'], risks,
                              plan['representations'], available,
                              self.budget - plan['same_plan_all_r_peak_bytes'],
                              self.risk_budget * microbatch_tokens / update_tokens)
        return self.select(recovery, actions)
