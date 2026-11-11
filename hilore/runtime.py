import json
import math
from pathlib import Path
from .allocator import allocate, warmup
from .objective import exposure
from .profile_schema import validate_profile


class Scheduler:
    def __init__(self, profile_path, configuration_fingerprint, memory_multiplier, risk_budget,
                 allow_development=False, action_transform=None, initializing=False):
        self.action_transform = action_transform
        self.initializing = initializing
        self.profile = profile_path if isinstance(profile_path, dict) else json.loads(Path(profile_path).read_text())
        self.budget = validate_profile(self.profile, configuration_fingerprint, memory_multiplier,
                                       risk_budget, allow_development, initializing)
        self.risk_budget = risk_budget
        self.plan = None
        warmup()

    def before_forward(self, shape, layer_count):
        self.shape = list(shape)
        candidates = [plan for plan in self.profile['plans'] if plan['shape'] == self.shape
                      and plan['estimated_same_plan_all_r_peak_bytes'] <= self.budget
                      and (self.initializing or plan['validated_peak_bytes'] <= self.budget)]
        self.plan = min(candidates, key=lambda plan: plan['estimated_microbatch_seconds'], default=None)
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
            if set(unit['actions']) != {'R'} and not unit['recomputation_verified']:
                raise ValueError('Unverified recovery unit in profile')
            actions = set(unit['actions']) & bundle.available
            available.append(actions)
            if 'L' in actions:
                calibration = self.profile['calibration'][str(i)]
                if calibration['group'] != f'mlp/{i}':
                    raise ValueError('Unsupported susceptibility group')
                if calibration.get('selected_index') != bundle.selected:
                    raise ValueError('Calibration refers to a different compressed component')


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
                              self.budget - plan['estimated_same_plan_all_r_peak_bytes'],
                              self.risk_budget * microbatch_tokens / update_tokens)
        return self.select(recovery, actions)
