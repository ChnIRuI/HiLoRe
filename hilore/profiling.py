from collections import Counter
import gc
import statistics
import time
import threading
import copy
import random
import numpy as np
import torch
import torch.distributed as dist


class Snapshot:


    def __init__(self, model, optimizer, recovery):
        self.model, self.optimizer, self.recovery = model, optimizer, recovery
        self.parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
        from torch.utils._pytree import tree_map
        self.optimizer_state = tree_map(lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x),
                                       optimizer.state_dict())
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        self.python_rng, self.numpy_rng = random.getstate(), np.random.get_state()

    def restore(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.recovery.bundles.clear()
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.parameters:
                    p.copy_(self.parameters[name])
            for name, b in self.model.named_buffers():
                b.copy_(self.buffers[name])

        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)

    def state_dict(self):
        kind, keys, position, gaussian, cached = self.numpy_rng
        return {'parameters': self.parameters, 'buffers': self.buffers,
                'optimizer': self.optimizer_state, 'cpu_rng': self.cpu_rng, 'cuda_rng': self.cuda_rng,
                'python_rng': self.python_rng,
                'numpy_rng': (kind, torch.from_numpy(keys.astype(np.int64)), position, gaussian, cached)}

    def load_state_dict(self, state):
        for key, expected in [('parameters', self.parameters), ('buffers', self.buffers)]:
            if state[key].keys() != expected.keys() or any(state[key][n].shape != v.shape for n, v in expected.items()):
                raise ValueError('Checkpoint shard topology does not match the current model')
        self.parameters, self.buffers = state['parameters'], state['buffers']
        self.optimizer_state = state['optimizer']
        self.cpu_rng, self.cuda_rng, self.python_rng = state['cpu_rng'], state['cuda_rng'], state['python_rng']
        kind, keys, position, gaussian, cached = state['numpy_rng']
        self.numpy_rng = (kind, keys.numpy().astype(np.uint32), position, gaussian, cached)
        self.restore()


class MemorySampler:


    def __enter__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        index = torch.cuda._get_nvml_device_index(torch.cuda.current_device())
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.peak, self.error = 0, None
        self.stop = threading.Event()
        self.sample()
        def poll():
            while not self.stop.wait(0.01):
                try:
                    self.sample()
                except Exception as exc:
                    self.error = exc
                    return
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        self.peak = max(self.peak, self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        try:
            self.sample()
        finally:
            self.nvml.nvmlShutdown()
        if self.error is not None:
            raise RuntimeError('NVML sampling failed') from self.error


def rank_max(values):
    tensor = torch.tensor(values, dtype=torch.float64, device='cuda')
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.tolist()


def recovery_footprint(events, unit_name):


    footprint = Counter()
    for event in events:
        parent = event.cpu_parent
        lineage = []
        while parent is not None:
            lineage.append(parent.name)
            parent = parent.cpu_parent
        if unit_name not in lineage or 'recovery_replay' not in lineage:
            continue
        if not event.name.startswith('aten::'):
            continue
        identity = (event.name, repr(event.input_shapes), tuple(event.stack), tuple(lineage))
        kernels = getattr(event, 'kernels', [])
        for kernel in kernels:
            footprint[(identity, kernel.name)] += 1
    return footprint


def verify_elimination(reference_events, materialized_events, unit_name):

    reference = recovery_footprint(reference_events, unit_name)
    candidate = recovery_footprint(materialized_events, unit_name)
    removed = reference - candidate
    if not removed:
        raise ValueError('No attributed recomputation kernels were eliminated')
    if candidate:
        raise ValueError('Materialized complete bundle still executes attributed replay kernels')
    return removed


def measure_execution(run, restore, representations, actions, repeats, warmups, scope):
    if scope not in ('microbatch', 'complete_update') or repeats < 1 or warmups < 0:
        raise ValueError('Invalid measurement scope or repetition count')
    for _ in range(warmups):
        restore()
        run(representations, actions)
    rows = []
    for _ in range(repeats):
        restore()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        with MemorySampler() as sampler:
            start = time.perf_counter()
            result = run(representations, actions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        complete = scope == 'complete_update'
        if result.get('complete_update') is not complete or not result.get('after_reduction_before_clipping'):
            raise ValueError('Execution boundary differs from the declared measurement scope')
        elapsed, peak, allocated, reserved = rank_max([
            elapsed, sampler.peak, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()])
        row = {'seconds': elapsed, 'peak_bytes': peak, 'allocated_bytes': allocated,
               'reserved_bytes': reserved}
        if complete:
            row['gradients'] = result['gradients']
        for key in ('actions', 'recovery_action_counts'):
            if key in result:
                row[key] = result[key]
        rows.append(row)
    return {'seconds': statistics.mean(row['seconds'] for row in rows),
            'peak_bytes': max(row['peak_bytes'] for row in rows), 'runs': rows,
            'measurement_scope': scope, 'repetitions': repeats, 'warmups': warmups}


def measure_microbatch(run, restore, representations, actions, repeats=2):
    if repeats < 2:
        raise ValueError('Microbatch measurements require at least two repetitions')
    return measure_execution(run, restore, representations, actions, repeats, 1, 'microbatch')


def measure_update_once(run, restore, representations, actions):
    return measure_execution(run, restore, representations, actions, 1, 0, 'complete_update')


def paired_measure(run, restore, representations, actions, repeats=5):
    if repeats < 2:
        raise ValueError('Paired update measurements require at least two repetitions')
    return measure_execution(run, restore, representations, actions, repeats, 1, 'complete_update')


def recomputation_seconds(events, unit_name):
    elapsed = 0.0
    for event in events:
        parent, lineage = event.cpu_parent, []
        while parent is not None:
            lineage.append(parent.name)
            parent = parent.cpu_parent
        if unit_name in lineage and 'recovery_replay' in lineage and event.name.startswith('aten::'):
            elapsed += sum(kernel.duration for kernel in getattr(event, 'kernels', []))
    return elapsed / 1e6


def conditional_costs(measure, representations, feasible_actions):

    all_r = ['R'] * len(representations)
    baseline = measure(representations, all_r)
    utility, memory, admitted = [], [], []
    for unit, actions in enumerate(feasible_actions):
        values, costs, available = [0., 0., 0.], [0., 0., 0.], ['R']
        for action in ('H', 'L'):
            if action not in actions:
                continue
            trial = all_r.copy()
            trial[unit] = action
            result = measure(representations, trial)
            benefit = baseline['seconds'] - result['seconds']
            if benefit > 0:
                index = 'RHL'.index(action)
                values[index] = benefit
                costs[index] = max(0., result['peak_bytes'] - baseline['peak_bytes'])
                available.append(action)
        utility.append(values)
        memory.append(costs)
        admitted.append(available)
    return baseline, utility, memory, admitted


def validate_joint(measure, representations, actions, memory_budget):

    actions = list(actions)
    while True:
        joint = measure(representations, actions)
        removed = False
        for unit, action in enumerate(actions):
            if action == 'R':
                continue
            trial = actions.copy()
            trial[unit] = 'R'
            alternative = measure(representations, trial)
            if alternative['seconds'] - joint['seconds'] <= 0:
                actions[unit] = 'R'
                removed = True
                break
        if not removed:
            break
    if joint['peak_bytes'] > memory_budget:
        raise ValueError('Joint schedule exceeds the measured GC-relative ceiling')
    return actions, joint
