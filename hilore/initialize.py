"""Measure decoder recovery plans and calibrate a development HiLoRe profile."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .calibration import fit
from .data import load_update, microbatches
from .objective import exposure, gradient_error
from .profiling import Snapshot, paired_measure, verify_elimination
from .runtime import Scheduler
from .splits import SplitManifest
from .train import (CONFIG_PATH, actor_update, backward_batch, build_model, fingerprint,
                    gradient_snapshot, make_optimizer, model_identity, setup_distributed,
                    validate_configuration)


def across_ranks(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def summed(values):
    tensor = torch.tensor(values, dtype=torch.float64, device='cuda')
    dist.all_reduce(tensor)
    return tensor.tolist()


class Fixed:
    """Preserve capture while changing recovery at one or all shape occurrences."""

    def __init__(self, representations, actions, shape=None, occurrence=None):
        self.q, self.a, self.shape, self.occurrence = representations, actions, shape, occurrence
        self.index = -1

    def before_forward(self, shape, layer_count):
        self.index += 1
        self.active = self.shape is None or tuple(shape) == tuple(self.shape)
        return self.q if self.active else ['R'] * layer_count

    def after_forward(self, recovery, omega, mask, tokens, total):
        active = self.active and (self.occurrence is None or self.index == self.occurrence)
        requested = self.a if active else ['R'] * len(recovery.layers)
        actual = recovery.select(requested)
        if actual != requested:
            raise RuntimeError('A measured recovery action fell back; reject this candidate')


def capture_metadata(recovery, omega, batch):
    result = []
    for bundle in recovery.bundles.values():
        result.append({'saved_state_signature': [[list(a), list(b), c] for a, b, c, _ in bundle.records],
                       'selected_index': bundle.selected,
                       'eligible': bundle.selected is not None and not bundle.invalid_alias,
                       'support_kind': 'complete_dense_decoder'})
    return result


def choose_capture(ranked, h_fraction, headroom, count):
    """Reserve measured capture headroom separately for H and L (Table 8)."""
    q = ['R'] * count
    h_left, l_left = headroom * h_fraction, headroom * (1 - h_fraction)
    for row in ranked:
        i = row['unit']
        if row['H'] is not None and row['H']['capture_cost'] <= h_left:
            q[i] = 'H'
            h_left -= row['H']['capture_cost']
        elif row['L'] is not None and row['L']['capture_cost'] <= l_left:
            q[i] = 'L'
            l_left -= row['L']['capture_cost']
    return q


class Initializer:
    def __init__(self, model, optimizer, recovery, config, device, manifest, files, repeats):
        self.model, self.optimizer, self.recovery = model, optimizer, recovery
        self.config, self.device, self.manifest = config, device, manifest
        self.files, self.repeats = files, repeats
        self.snapshot = Snapshot(model, optimizer, recovery)
        self.count = len(recovery.layers)
        self.all_r = ['R'] * self.count
        self.data = None

    def batches(self, data):
        return list(microbatches(data, dist.get_rank(), dist.get_world_size(),
                                self.config['microbatch_token_target_per_rank']))

    def restore(self):
        self.snapshot.restore()
        self.recovery.measure_error = False

    def update(self, q, actions, shape=None, occurrence=None, gc=False):
        self.recovery.method = 'gc' if gc else 'hilore'
        scheduler = None if gc else Fixed(q, actions, shape, occurrence)
        return actor_update(self.model, self.optimizer, self.recovery, scheduler,
                            self.data, self.config, self.device, collect_gradients=True)

    def measure(self, q, actions, shape=None, occurrence=None, gc=False):
        return paired_measure(lambda p, a: self.update(p, a, shape, occurrence, gc),
                              self.restore, q, actions, self.repeats)

    def batch(self, batch, q, actions, observer=None):
        self.recovery.method = 'hilore'
        backward_batch(self.model, self.recovery, Fixed(q, actions), batch,
                       int(self.data['response_mask'].sum()), self.config, self.device, observer)
        return gradient_snapshot(self.model)

    def graph(self, batch):
        metadata = []
        self.restore()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as reference:
            self.batch(batch, self.all_r, self.all_r,
                       lambda r, w, b: metadata.extend(capture_metadata(r, w, b)))
        gathered = across_ranks(metadata)
        if any([[u['saved_state_signature'], u['selected_index']] for u in other] !=
               [[u['saved_state_signature'], u['selected_index']] for u in metadata] for other in gathered):
            raise ValueError('Rank-dependent saved-state topology is unsupported')
        for unit, entry in enumerate(metadata):
            entry['actions'] = ['R']
            entry['recomputation_verified'] = True
            for action in ('H', 'L'):
                if action == 'L' and not all(rows[unit]['eligible'] for rows in gathered):
                    continue
                q, actions = self.all_r.copy(), self.all_r.copy()
                q[unit] = actions[unit] = action
                self.restore()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                          torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as candidate:
                    self.batch(batch, q, actions)
                verified = True
                try:
                    removed = verify_elimination(reference.events(), candidate.events(), f'unit/{unit}')
                except ValueError:
                    verified = False
                    removed = {}
                if all(across_ranks(verified)):
                    entry['actions'].append(action)
                    entry[f'{action}_removed_kernel_count'] = min(across_ranks(sum(removed.values())))
        return metadata

    def calibrate(self, samples, eligible):
        rows = []
        for sample_id, (file, batch_index) in enumerate(samples):
            self.data = load_update(file, self.config, self.manifest)
            batch = self.batches(self.data)[batch_index]
            groups = set(batch['prompt_ids'].tolist())
            ids = [p for g, p in zip(self.data['prompt_ids'].tolist(), self.data['problem_ids']) if g in groups]
            ids = sorted(set(p for values in across_ranks(ids) for p in values))
            self.restore()
            exact = self.batch(batch, self.all_r, self.all_r)
            for unit in eligible:
                self.restore()
                self.recovery.measure_error = True
                q = self.all_r.copy()
                q[unit] = 'L'
                observation = {}
                def observe(recovery, omega, tensors):
                    bundle = recovery.bundles[unit]
                    numerator, denominator, exp = summed([
                        bundle.error_numerator, bundle.state_squared_norm,
                        exposure(omega, tensors['response_mask'].bool())])
                    observation.update(reconstruction_error=math.sqrt(numerator) / (math.sqrt(denominator) + 1e-8),
                                       exposure=exp, selected_index=bundle.selected)
                candidate = self.batch(batch, q, q, observe)
                rows.append(dict(observation, unit=unit, group=f'mlp/{unit}', microbatch=sample_id,
                    split='train', intervention_count=1, paired_state_verified=True,
                    problem_ids=ids, split_manifest_sha256=self.manifest.sha256,
                    gradient_error=gradient_error(exact, candidate, distributed=True)))
        calibrated = fit(rows, self.manifest) if rows else {}
        for unit in eligible:
            indices = {row['selected_index'] for row in rows if row['unit'] == unit}
            if len(indices) != 1:
                raise ValueError('The compressed component changed across calibration microbatches')
            calibrated[str(unit)]['selected_index'] = next(iter(indices))
        return calibrated

    def profile_shape(self, shape, file, batch_index, units, calibration, gc_peak, budget):
        self.data = load_update(file, self.config, self.manifest)
        ranked = []
        for unit, entry in enumerate(units):
            row = {'unit': unit, 'H': None, 'L': None, 'score': 0.0}
            for action in ('H', 'L'):
                if action not in entry['actions'] or (action == 'L' and (str(unit) not in calibration
                        or calibration[str(unit)]['selected_index'] != entry['selected_index'])):
                    continue
                q, a = self.all_r.copy(), self.all_r.copy()
                q[unit] = a[unit] = action
                baseline = self.measure(q, self.all_r, shape)
                candidate = self.measure(q, a, shape, batch_index)
                benefit = baseline['seconds'] - candidate['seconds']
                cost = max(0.0, max(baseline['peak_bytes'], candidate['peak_bytes']) - gc_peak)
                if benefit > 0:
                    row[action] = {'capture_cost': cost, 'benefit': benefit}
                    row['score'] = max(row['score'], benefit / max(cost, 1.0))
            ranked.append(row)
        ranked.sort(key=lambda row: (-row['score'], row['unit']))
        plans, seen = [], set()
        for fraction in (0.0, 0.5, 1.0):
            q = choose_capture(ranked, fraction, max(0.0, budget - gc_peak), self.count)
            if tuple(q) in seen or q == self.all_r:
                continue
            seen.add(tuple(q))
            baseline = self.measure(q, self.all_r, shape)
            if baseline['peak_bytes'] > budget:
                continue
            utility, memory, admitted = [], [], copy.deepcopy(units)
            for unit in range(self.count):
                values, costs, actions = [0.0] * 3, [0.0] * 3, ['R']
                for action in ('H', 'L'):
                    if (action not in units[unit]['actions'] or action not in {'H': 'RHL', 'L': 'RL', 'R': 'R'}[q[unit]]
                            or (action == 'L' and (str(unit) not in calibration
                                or calibration[str(unit)]['selected_index'] != units[unit]['selected_index']))):
                        continue
                    a = self.all_r.copy()
                    a[unit] = action
                    trial = self.measure(q, a, shape, batch_index)
                    benefit = baseline['seconds'] - trial['seconds']
                    if benefit > 0:
                        j = 'RHL'.index(action)
                        values[j] = benefit
                        costs[j] = max(0.0, trial['peak_bytes'] - baseline['peak_bytes'])
                        actions.append(action)
                admitted[unit]['actions'] = actions
                utility.append(values)
                memory.append(costs)
            plans.append({'shape': list(shape), 'representations': q, 'units': admitted,
                          'utility_seconds': utility, 'marginal_peak_bytes': memory,
                          'same_plan_all_r_peak_bytes': baseline['peak_bytes'],
                          'same_plan_all_r_seconds': baseline['seconds'],
                          'validated_peak_bytes': baseline['peak_bytes'],
                          'validated_actor_update_seconds': baseline['seconds'],
                          'capture_h_fraction': fraction})
        return plans

    def validate(self, profile, identity):
        """Measure the actual dynamic scheduler, then remove nonpositive actions."""
        while True:
            joint = self.dynamic_measure(profile, identity)
            removed = False
            for plan_index, plan in enumerate(profile['plans']):
                for unit, entry in enumerate(plan['units']):
                    if len(entry['actions']) == 1:
                        continue
                    # Run the identical allocator, then override only this unit.
                    # Reallocating its released budget would not be leave-one-out.
                    trial = self.dynamic_measure(profile, identity, (plan['shape'], unit))
                    if trial['seconds'] - joint['seconds'] <= 0:
                        entry['actions'] = ['R']
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
        return joint

    def dynamic_measure(self, profile, identity, omit=None):
        def transform(shape, actions):
            if omit is not None and shape == omit[0]:
                actions[omit[1]] = 'R'
            return actions
        scheduler = Scheduler(profile, identity, self.config['memory_multiplier'], profile['risk_budget'], True,
                              action_transform=transform)
        def run(q, a):
            self.recovery.method = 'hilore'
            self.last_schedule = []
            def record(recovery, omega, batch):
                self.last_schedule.append({'shape': list(batch['input_ids'].shape),
                    'q': [recovery.bundles[i].representation for i in range(self.count)],
                    'a': [recovery.bundles[i].action for i in range(self.count)]})
            return actor_update(self.model, self.optimizer, self.recovery, scheduler,
                                self.data, self.config, self.device, collect_gradients=True, observer=record)
        return paired_measure(run, self.restore, [], [], self.repeats)

    def evaluate(self, profile, identity, references, prune=True):
        """Validate the final joint schedule across all complete development updates."""
        if prune:
            while True:
                before = json.dumps([p['units'] for p in profile['plans']], sort_keys=True)
                for path in self.files:
                    self.data = load_update(path, self.config, self.manifest)
                    self.validate(profile, identity)
                if before == json.dumps([p['units'] for p in profile['plans']], sort_keys=True):
                    break
        results = []
        for path in self.files:
            self.data = load_update(path, self.config, self.manifest)
            measured = self.dynamic_measure(profile, identity)
            errors = [gradient_error(ref['gradients'], cand['gradients'], True)
                      for ref, cand in zip(references[path]['runs'], measured['runs'])]
            results.append({'peak_bytes': measured['peak_bytes'], 'seconds': measured['seconds'],
                            'gradient_error': sum(errors) / len(errors),
                            'allocated_peak_bytes': max(r['allocated_bytes'] for r in measured['runs']),
                            'reserved_peak_bytes': max(r['reserved_bytes'] for r in measured['runs'])})
        return results


def development_acceptable(rows, budget):
    return (bool(rows) and math.isfinite(budget) and budget > 0
            and all(math.isfinite(row[k]) and row[k] >= 0 for row in rows
                    for k in ('peak_bytes', 'gradient_error', 'seconds'))
            and max(r['peak_bytes'] for r in rows) <= budget
            and sum(r['gradient_error'] for r in rows) / len(rows) <= .015)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--model')
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--updates', type=Path, required=True, help='Development actor replay directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--risk-budget', type=float, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_configuration(config)
    if int(os.environ.get('WORLD_SIZE', '1')) != 4 or args.repeats < 5:
        parser.error('Initialization requires four CUDA ranks and at least five paired repetitions')
    if not math.isfinite(args.risk_budget) or args.risk_budget < 0:
        parser.error('Risk budget must be finite and nonnegative')
    if args.output.exists():
        parser.error('Output must be a new directory')
    config.update(method='hilore', risk_budget=args.risk_budget)
    if args.model:
        config['model'] = args.model
    manifest = SplitManifest(args.split_manifest)
    config['split_manifest_sha256'] = manifest.sha256
    files = sorted(args.updates.glob('update_*.pt'))
    if not files:
        parser.error('Provide measured development replay inputs')
    for path in files:
        load_update(path, config, manifest)
    identity = model_identity(config['model'])
    device = setup_distributed(config)
    model, recovery = build_model(config['model'], config, device, identity)
    optimizer = make_optimizer(model, config)
    init = Initializer(model, optimizer, recovery, config, device, manifest, files, args.repeats)
    signature = fingerprint(config, identity)
    try:
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        shapes, samples_by_shape, references = {}, {}, {}
        for path in files:
            init.data = load_update(path, config, manifest)
            for index, batch in enumerate(init.batches(init.data)):
                shape = tuple(batch['input_ids'].shape)
                shapes.setdefault(shape, (path, index))
                samples_by_shape.setdefault(shape, []).append((path, index))
            references[path] = init.measure(init.all_r, init.all_r, gc=True)
        gc_peak = max(result['peak_bytes'] for result in references.values())
        budget = gc_peak * config['memory_multiplier']
        graphs = {}
        for shape, (path, index) in shapes.items():
            init.data = load_update(path, config, manifest)
            graphs[shape] = init.graph(init.batches(init.data)[index])
        supported = [shape for shape in shapes if len(samples_by_shape[shape]) >= 16]
        if not supported:
            raise ValueError('Supply at least 16 distinct training microbatches in one supported shape class')
        calibration_shape = max(supported, key=lambda shape: (
            sum('L' in unit['actions'] for unit in graphs[shape]), math.prod(shape), shape))
        samples = samples_by_shape[calibration_shape][:16]
        eligible = [i for i, unit in enumerate(graphs[calibration_shape]) if 'L' in unit['actions']]
        calibration = init.calibrate(samples, eligible)
        base = {'schema_version': 2, 'configuration_fingerprint': signature,
                'model_identity': identity, 'split_manifest_sha256': manifest.sha256,
                'memory_metric': 'nvml_device_used_max', 'measurement_world_size': 4,
                'memory_multiplier': config['memory_multiplier'], 'measured_gc_peak_bytes': gc_peak,
                'risk_budget': args.risk_budget, 'calibration': calibration,
                'calibration_shape': list(calibration_shape),
                'joint_validation_passed': True, 'terminal_quality_passed': False,
                'status': 'development_only',
                'replay_sha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]}
        selected = []
        all_r_results = init.evaluate(dict(base, plans=[]), signature, references, prune=False)
        all_r_seconds = sum(row['seconds'] for row in all_r_results)
        for shape, (path, index) in shapes.items():
            candidates = init.profile_shape(shape, path, index, graphs[shape], calibration, gc_peak, budget)
            accepted = []
            for plan in candidates:
                trial = dict(base, plans=[copy.deepcopy(plan)])
                measured = init.evaluate(trial, signature, references)
                seconds = sum(row['seconds'] for row in measured)
                if development_acceptable(measured, budget) and seconds < all_r_seconds:
                    plan = trial['plans'][0]
                    plan.update(validated_peak_bytes=max(r['peak_bytes'] for r in measured),
                                validated_actor_update_seconds=seconds / len(measured))
                    accepted.append(plan)
            if accepted:
                selected.append(min(accepted, key=lambda p: p['validated_actor_update_seconds']))
        profile = dict(base, plans=selected)
        results = init.evaluate(profile, signature, references)
        if not development_acceptable(results, budget) or sum(r['seconds'] for r in results) >= all_r_seconds:
            profile['plans'] = []
            selected = []
            results = all_r_results
        profile['development_measurements'] = results
        profile['joint_validation_passed'] = development_acceptable(results, budget)
        if not profile['joint_validation_passed']:
            profile['status'] = 'rejected'
        if dist.get_rank() == 0:
            (args.output / 'profile.json').write_text(json.dumps(profile, indent=2, allow_nan=False) + '\n')
            print(json.dumps({'status': profile['status'], 'plans': len(selected)}), flush=True)
    finally:
        init.restore()
        recovery.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
