import argparse
import copy
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from .calibration import fit
from .data import load_update, microbatches
from .objective import exposure, gradient_error
from .profile_schema import choose_capture, development_acceptable, shortlist, validate_profile
from .profiling import (Snapshot, measure_microbatch, measure_update_once, rank_max,
                        recomputation_seconds, verify_elimination)
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


class InitializationRejected(RuntimeError):
    pass


class Fixed:
    def __init__(self, representations, actions):
        self.q, self.a = representations, actions

    def before_forward(self, shape, layer_count):
        return self.q

    def after_forward(self, recovery, omega, mask, tokens, total):
        actual = recovery.select(self.a)
        if actual != self.a:
            raise InitializationRejected('A measured recovery action fell back')


def capture_metadata(recovery, omega, batch):
    result = []
    for index in range(len(recovery.layers)):
        bundle = recovery.bundles[index]
        storage_sizes = {}
        for key, (shape, stride, dtype, device) in zip(bundle.storage_keys, bundle.records):
            if key in bundle.checkpoint_storage:
                continue
            element_size = torch.empty((), dtype=getattr(torch, dtype.split('.')[-1])).element_size()
            span = 0 if any(size == 0 for size in shape) else 1 + sum((size - 1) * step for size, step in zip(shape, stride))
            storage_sizes[key] = max(storage_sizes.get(key, 0), span * element_size)
        h_bytes = sum(storage_sizes.values())
        eligible = bundle.selected is not None and not bundle.invalid_alias
        l_bytes = None
        if eligible:
            shape = bundle.records[bundle.selected][0]
            l_bytes = h_bytes - math.prod(shape)
        result.append({'saved_state_signature': [[list(a), list(b), c] for a, b, c, _ in bundle.records],
                       'selected_index': bundle.selected, 'eligible': eligible,
                       'capture_bytes_estimate': {'H': h_bytes, 'L': l_bytes},
                       'support_kind': 'complete_dense_decoder', 'actions': ['R'],
                       'recomputation_verified': False})
    return result


class Initializer:
    def __init__(self, model, optimizer, recovery, config, device, manifest, data,
                 repeats, candidate_limit, interaction_rounds):
        self.model, self.optimizer, self.recovery = model, optimizer, recovery
        self.config, self.device, self.manifest = config, device, manifest
        self.data, self.repeats = data, repeats
        self.candidate_limit, self.interaction_rounds = candidate_limit, interaction_rounds
        self.snapshot = Snapshot(model, optimizer, recovery)
        self.count = len(recovery.layers)
        self.all_r = ['R'] * self.count
        self.batches = list(microbatches(data, dist.get_rank(), dist.get_world_size(),
                                       config['microbatch_token_target_per_rank']))
        self.total_tokens = int(data['response_mask'].sum())
        self.timings = {}
        self.executions = {'microbatch': 0, 'complete_gc_update': 0, 'complete_hilore_update': 0}

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.timings[name] = rank_max([time.perf_counter() - started])[0]

    def restore(self):
        self.snapshot.restore()
        self.recovery.measure_error = False

    def batch(self, batch, q, actions, observer=None, collect_gradients=False, gc=False):
        self.executions['microbatch'] += 1
        self.recovery.method = 'gc' if gc else 'hilore'
        scheduler = None if gc else Fixed(q, actions)
        backward_batch(self.model, self.recovery, scheduler, batch, self.total_tokens,
                       self.config, self.device, observer)
        result = {'complete_update': False, 'after_reduction_before_clipping': True}
        if collect_gradients:
            result['gradients'] = gradient_snapshot(self.model)
        return result

    def measure_batch(self, batch, q, actions, gc=False):
        return measure_microbatch(lambda p, a: self.batch(batch, p, a, gc=gc),
                                  self.restore, q, actions, self.repeats)

    def complete_update(self, identity, profile=None):
        scheduler = None if profile is None else self.scheduler(profile, identity)
        def run(q, actions):
            gc = profile is None
            self.executions['complete_gc_update' if gc else 'complete_hilore_update'] += 1
            self.recovery.method = 'gc' if gc else 'hilore'
            counts = dict.fromkeys('RHL', 0)
            def observe(recovery, omega, tensors):
                for bundle in recovery.bundles.values():
                    counts[bundle.action] += 1
            result = actor_update(self.model, self.optimizer, self.recovery, scheduler,
                                  self.data, self.config, self.device, collect_gradients=True,
                                  observer=None if gc else observe)
            if not gc:
                result['recovery_action_counts'] = counts
            return result
        return measure_update_once(run, self.restore, [], [])

    def graph(self, batch, headroom):
        metadata = []
        self.restore()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as reference:
            self.batch(batch, self.all_r, self.all_r,
                       lambda recovery, omega, tensors: metadata.extend(capture_metadata(recovery, omega, tensors)))
        gathered = across_ranks(metadata)
        topology = [[row['saved_state_signature'], row['selected_index']] for row in metadata]
        if len(metadata) != self.count or any(
                [[row['saved_state_signature'], row['selected_index']] for row in rows] != topology
                for rows in gathered):
            raise ValueError('Rank-dependent saved-state topology is unsupported')
        durations = across_ranks([recomputation_seconds(reference.events(), f'unit/{i}')
                                 for i in range(self.count)])
        for index, entry in enumerate(metadata):
            entry['eligible'] = all(rows[index]['eligible'] for rows in gathered)
            entry['capture_bytes_estimate']['H'] = max(rows[index]['capture_bytes_estimate']['H'] for rows in gathered)
            entry['capture_bytes_estimate']['L'] = (max(rows[index]['capture_bytes_estimate']['L'] for rows in gathered)
                                                     if entry['eligible'] else None)
            entry['recomputation_seconds_estimate'] = max(values[index] for values in durations)
        candidates = shortlist(metadata, self.candidate_limit, headroom)
        for unit in candidates:
            entry = metadata[unit]
            for action in ('H', 'L'):
                if action == 'L' and not entry['eligible']:
                    continue
                q = self.all_r.copy()
                q[unit] = action
                self.restore()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as candidate:
                    self.batch(batch, q, q)
                try:
                    removed = verify_elimination(reference.events(), candidate.events(), f'unit/{unit}')
                    verified = True
                except ValueError:
                    removed, verified = {}, False
                checks = across_ranks((verified, sum(removed.values())))
                if all(check[0] for check in checks):
                    entry['actions'].append(action)
                    entry['recomputation_verified'] = True
                    entry[f'{action}_removed_kernel_count'] = min(check[1] for check in checks)
        if not any(len(row['actions']) > 1 for row in metadata):
            raise InitializationRejected('No shortlisted recovery action passed graph verification')
        return metadata, candidates

    def calibrate(self, indices, eligible):
        rows = []
        for sample_id, batch_index in enumerate(indices):
            if not eligible:
                break
            batch = self.batches[batch_index]
            groups = set(batch['prompt_ids'].tolist())
            ids = [problem for group, problem in zip(self.data['prompt_ids'].tolist(), self.data['problem_ids']) if group in groups]
            ids = sorted(set(problem for values in across_ranks(ids) for problem in values))
            self.restore()
            exact = self.batch(batch, self.all_r, self.all_r, collect_gradients=True)['gradients']
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
                candidate = self.batch(batch, q, q, observe, collect_gradients=True)['gradients']
                rows.append(dict(observation, unit=unit, group=f'mlp/{unit}', microbatch=sample_id,
                    split='train', intervention_count=1, paired_state_verified=True,
                    problem_ids=ids, split_manifest_sha256=self.manifest.sha256,
                    gradient_error=gradient_error(exact, candidate, distributed=True)))
        calibrated = fit(rows, self.manifest) if rows else {}
        for unit in eligible:
            selected = {row['selected_index'] for row in rows if row['unit'] == unit}
            if len(selected) != 1:
                raise ValueError('Compressed component changed across calibration microbatches')
            calibrated[str(unit)]['selected_index'] = next(iter(selected))
        return calibrated

    def profile_shape(self, batch, units, calibration, gc_peak, budget, micro_gc):
        ranked = []
        for unit, entry in enumerate(units):
            row = {'unit': unit, 'H': None, 'L': None, 'score': 0.0}
            for action in ('H', 'L'):
                if action not in entry['actions'] or (action == 'L' and (str(unit) not in calibration
                        or calibration[str(unit)]['selected_index'] != entry['selected_index'])):
                    continue
                cost = entry['capture_bytes_estimate'][action]
                if cost is not None and cost > 0:
                    benefit = entry['recomputation_seconds_estimate']
                    row[action] = {'capture_cost': cost, 'benefit': benefit}
                    row['score'] = max(row['score'], benefit / cost)
            ranked.append(row)
        ranked.sort(key=lambda row: (-row['score'], row['unit']))
        plans, seen = [], set()
        for fraction in (0.0, 0.5, 1.0):
            q = choose_capture(ranked, fraction, max(0.0, budget - gc_peak), self.count)
            if tuple(q) in seen or q == self.all_r:
                continue
            seen.add(tuple(q))
            baseline = self.measure_batch(batch, q, self.all_r)
            estimated_peak = gc_peak + max(0.0, baseline['peak_bytes'] - micro_gc['peak_bytes'])
            if estimated_peak > budget:
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
                    trial = self.measure_batch(batch, q, a)
                    benefit = baseline['seconds'] - trial['seconds']
                    if benefit > 0:
                        index = 'RHL'.index(action)
                        values[index] = benefit
                        costs[index] = max(0.0, trial['peak_bytes'] - baseline['peak_bytes'])
                        actions.append(action)
                admitted[unit]['actions'] = actions
                utility.append(values)
                memory.append(costs)
            if any(len(unit['actions']) > 1 for unit in admitted):
                plans.append({'shape': list(batch['input_ids'].shape), 'representations': q, 'units': admitted,
                    'utility_seconds': utility, 'marginal_peak_bytes': memory,
                    'measurement_scope': 'microbatch',
                    'estimated_same_plan_all_r_peak_bytes': estimated_peak,
                    'microbatch_all_r_peak_bytes': baseline['peak_bytes'],
                    'microbatch_all_r_seconds': baseline['seconds'],
                    'estimated_microbatch_seconds': baseline['seconds'],
                    'capture_h_fraction': fraction, 'interaction_screening_passed': False})
        if not plans:
            raise InitializationRejected('No candidate plan has positive measured utility within the estimated memory ceiling')
        return plans

    def scheduler(self, profile, identity, omit=None):
        def transform(shape, actions):
            if omit is not None and tuple(shape) == tuple(profile['plans'][0]['shape']):
                actions[omit] = 'R'
            return actions
        return Scheduler(profile, identity, self.config['memory_multiplier'], profile['risk_budget'],
                         allow_development=True, action_transform=transform, initializing=True)

    def dynamic_batch(self, batch, profile, identity, omit=None):
        scheduler = self.scheduler(profile, identity, omit)
        def run(q, actions):
            self.executions['microbatch'] += 1
            self.recovery.method = 'hilore'
            actual = []
            def observe(recovery, omega, tensors):
                actual.extend(recovery.bundles[i].action for i in range(self.count))
            backward_batch(self.model, self.recovery, scheduler, batch, self.total_tokens,
                           self.config, self.device, observe)
            return {'complete_update': False, 'after_reduction_before_clipping': True, 'actions': actual}
        measured = measure_microbatch(run, self.restore, [], [], self.repeats)
        local = sorted({i for row in measured['runs'] for i, action in enumerate(row['actions']) if action != 'R'})
        measured['active_units'] = sorted({i for indices in across_ranks(local) for i in indices})
        return measured

    def select_plan(self, batch, base, plans, identity, micro_gc):
        candidates = []
        for plan in plans:
            profile = dict(base, plans=[copy.deepcopy(plan)])
            measured = self.dynamic_batch(batch, profile, identity)
            estimated_peak = base['measured_gc_peak_bytes'] + max(0.0, measured['peak_bytes'] - micro_gc['peak_bytes'])
            if measured['active_units'] and estimated_peak <= base['measured_gc_peak_bytes'] * self.config['memory_multiplier']:
                candidates.append((measured['seconds'], profile, measured))
        if not candidates:
            raise InitializationRejected('No candidate plan executes an admitted recovery action within the estimated ceiling')
        _, profile, joint = min(candidates, key=lambda row: row[0])
        profile['plans'][0]['estimated_microbatch_seconds'] = joint['seconds']
        return profile, joint

    def validate_interactions(self, batch, profile, identity, joint):
        plan = profile['plans'][0]
        for round_index in range(self.interaction_rounds):
            if round_index:
                joint = self.dynamic_batch(batch, profile, identity)
            if not joint['active_units']:
                raise InitializationRejected('Interaction screening left no active recovery actions')
            removed = []
            for unit in joint['active_units']:
                alternative = self.dynamic_batch(batch, profile, identity, unit)
                if alternative['seconds'] <= joint['seconds']:
                    removed.append(unit)
            if not removed:
                plan['interaction_screening_passed'] = True
                plan['interaction_rounds'] = round_index + 1
                plan['estimated_microbatch_seconds'] = joint['seconds']
                return
            for unit in removed:
                plan['units'][unit]['actions'] = ['R']
                plan['utility_seconds'][unit] = [0.0] * 3
                plan['marginal_peak_bytes'][unit] = [0.0] * 3
        raise InitializationRejected('Interaction screening did not converge within the configured round limit')


def make_parser():
    parser = argparse.ArgumentParser(description='Build a bounded microbatch profile with complete-update validation')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--model')
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--updates', type=Path, required=True)
    parser.add_argument('--validation-record')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--risk-budget', type=float, required=True)
    parser.add_argument('--candidate-limit', type=int, default=8)
    parser.add_argument('--microbatch-repeats', type=int, default=2)
    parser.add_argument('--interaction-rounds', type=int, default=2)
    return parser


def main():
    started = time.perf_counter()
    parser = make_parser()
    args = parser.parse_args()
    if (int(os.environ.get('WORLD_SIZE', '1')) != 4 or args.candidate_limit < 1
            or args.microbatch_repeats < 2 or args.interaction_rounds < 1):
        parser.error('Use four ranks, a positive candidate/round limit, and at least two microbatch repetitions')
    if not math.isfinite(args.risk_budget) or args.risk_budget < 0:
        parser.error('Risk budget must be finite and nonnegative')
    if args.output.exists():
        parser.error('Output must be a new directory')
    config = json.loads(args.config.read_text())
    validate_configuration(config)
    config.update(method='hilore', risk_budget=args.risk_budget)
    if args.model:
        config['model'] = args.model
    manifest = SplitManifest(args.split_manifest)
    config['split_manifest_sha256'] = manifest.sha256
    files = sorted(args.updates.glob('update_*.pt'))
    if args.validation_record is not None:
        files = [path for path in files if path.name == args.validation_record]
    if not files:
        parser.error('Provide a complete development replay, optionally selected by filename')
    path = files[0]
    data = load_update(path, config, manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = model_identity(config['model'])
    device = setup_distributed(config)
    model, recovery = build_model(config['model'], config, device, identity)
    optimizer = make_optimizer(model, config)
    init = Initializer(model, optimizer, recovery, config, device, manifest, data,
                       args.microbatch_repeats, args.candidate_limit, args.interaction_rounds)
    signature = fingerprint(config, identity)
    profile = {'schema_version': 3, 'status': 'initializing', 'joint_validation_passed': False,
               'terminal_quality_passed': False, 'configuration_fingerprint': signature,
               'model_identity': identity, 'split_manifest_sha256': manifest.sha256,
               'memory_metric': 'nvml_device_used_max', 'measurement_world_size': 4,
               'memory_multiplier': config['memory_multiplier'], 'risk_budget': args.risk_budget,
               'plans': [], 'replay_sha256': [digest],
               'initialization': {'protocol': 'bounded_microbatch_v1', 'candidate_limit': args.candidate_limit,
                    'calibration_microbatches': 16, 'microbatch_repeats': args.microbatch_repeats,
                    'microbatch_warmups': 1, 'interaction_round_limit': args.interaction_rounds,
                    'complete_update_pairs': 1, 'complete_update_warmups': 0,
                    'validation_record': path.name, 'validation_record_sha256': digest,
                    'validation_record_selection': 'explicit' if args.validation_record else 'lexicographic_first'}}
    rejected = False
    try:
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        init.timings['load_and_setup'] = rank_max([time.perf_counter() - started])[0]
        shapes = {}
        for index, batch in enumerate(init.batches):
            shapes.setdefault(tuple(batch['input_ids'].shape), []).append(index)
        supported = [shape for shape, indices in shapes.items() if len(indices) >= 16]
        if not supported:
            raise InitializationRejected('The selected update must contain 16 microbatches in one shape class')
        shape = max(supported, key=lambda key: (len(shapes[key]), math.prod(key), key))
        indices = shapes[shape][:16]
        batch = init.batches[indices[0]]
        profile['calibration_shape'] = list(shape)
        profile['initialization']['calibration_indices'] = indices
        profile['initialization']['microbatches_per_rank'] = len(init.batches)
        profile['initialization']['profiled_shape_occurrences'] = len(shapes[shape])
        with init.stage('microbatch_reference_and_warmup'):
            micro_gc = init.measure_batch(batch, init.all_r, init.all_r, gc=True)
        with init.stage('complete_gc_reference'):
            reference = init.complete_update(signature)
        gc_peak = reference['peak_bytes']
        budget = gc_peak * config['memory_multiplier']
        profile['measured_gc_peak_bytes'] = gc_peak
        with init.stage('graph_screening'):
            units, candidates = init.graph(batch, max(0.0, budget - gc_peak))
        profile['initialization']['shortlisted_units'] = candidates
        eligible = [i for i, unit in enumerate(units) if 'L' in unit['actions']]
        profile['initialization']['calibrated_units'] = eligible
        with init.stage('calibration'):
            profile['calibration'] = init.calibrate(indices, eligible)
        with init.stage('microbatch_utility_profiling'):
            plans = init.profile_shape(batch, units, profile['calibration'], gc_peak, budget, micro_gc)
        profile['initialization']['candidate_plan_count'] = len(plans)
        with init.stage('plan_selection'):
            profile, joint = init.select_plan(batch, profile, plans, signature, micro_gc)
        with init.stage('interaction_screening'):
            init.validate_interactions(batch, profile, signature, joint)
        with init.stage('complete_hilore_validation'):
            measured = init.complete_update(signature, profile)
            counts = measured['runs'][0]['recovery_action_counts']
            totals = summed([counts[action] for action in 'RHL'])
            action_counts = dict(zip('RHL', map(int, totals)))
            row = {'measurement_scope': 'complete_update', 'repetitions': 1,
                   'recovery_action_counts': action_counts,
                   'gc_seconds': reference['seconds'], 'seconds': measured['seconds'],
                   'peak_bytes': measured['peak_bytes'],
                   'gradient_error': gradient_error(reference['runs'][0]['gradients'], measured['runs'][0]['gradients'], True),
                   'allocated_peak_bytes': measured['runs'][0]['allocated_bytes'],
                   'reserved_peak_bytes': measured['runs'][0]['reserved_bytes'],
                   'gain_percent': 100 * (reference['seconds'] / measured['seconds'] - 1)}
            profile['development_measurements'] = [row]
            if not action_counts['H'] + action_counts['L']:
                raise InitializationRejected('Complete-update validation used only R recovery')
            if not development_acceptable([row], budget, config['gradient_error_threshold']):
                raise InitializationRejected('Complete-update memory or gradient-fidelity validation failed')
            if measured['seconds'] >= reference['seconds']:
                raise InitializationRejected('The selected plan did not improve complete-update throughput')
            profile['plans'][0].update(validation_scope='complete_update', validated_peak_bytes=measured['peak_bytes'],
                                       validated_actor_update_seconds=measured['seconds'])
            profile.update(joint_validation_passed=True, status='development_only')
            validate_profile(profile, signature, config['memory_multiplier'], args.risk_budget, allow_development=True)
    except InitializationRejected as error:
        rejected = True
        profile.update(status='rejected', joint_validation_passed=False, rejection_reason=str(error))
    except Exception:
        profile.update(status='failed', joint_validation_passed=False)
        raise
    finally:
        with init.stage('cleanup'):
            init.restore()
            recovery.close()
        profile['timing_seconds'] = dict(init.timings)
        profile['timing_seconds']['total_wall'] = rank_max([time.perf_counter() - started])[0]
        profile['execution_counts'] = dict(init.executions)
        try:
            if dist.get_rank() == 0 and args.output.is_dir():
                destination = args.output / 'profile.json'
                temporary = args.output / 'profile.json.tmp'
                temporary.write_text(json.dumps(profile, indent=2, allow_nan=False) + '\n')
                temporary.replace(destination)
                print(json.dumps({'status': profile['status'], 'plans': len(profile['plans']),
                                  'wall_seconds': profile['timing_seconds']['total_wall'],
                                  'execution_counts': profile['execution_counts']}), flush=True)
        finally:
            dist.destroy_process_group()
    if rejected:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
