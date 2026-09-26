"""Deferred non-reentrant recovery of complete decoder state bundles."""

import types
import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_map, tree_flatten
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


class _StopReplay(Exception):
    pass


def _metadata(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _storage(tensor):
    return (str(tensor.device), StorageWeakRef(tensor.untyped_storage()))


class Bundle:
    """Keep the original autograd graph and defer saved-state recovery.

    Every saved tensor in the region has a handle. R repopulates all handles
    from exact checkpoint inputs under the original RNG and autocast settings.
    H/L supply handles directly and execute no region replay. The implementation
    supports one backward per forward, without retained or higher-order graphs.
    """

    def __init__(self, fn, args, kwargs, representation, device, minimum_bytes=32 << 20):
        if representation not in ('R', 'H', 'L'):
            raise ValueError("Unknown candidate representation")
        self.fn, self.representation = fn, representation
        self.args = tree_map(lambda x: x.detach().requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x, args)
        self.kwargs = tree_map(lambda x: x.detach().requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x, kwargs)
        self.device, self.minimum_bytes = device, minimum_bytes
        tensors = [x for x in tree_flatten((self.args, self.kwargs))[0] if isinstance(x, torch.Tensor)]
        self.checkpoint_storage = {_storage(x) for x in tensors}
        self.input_versions = [(x, x._version) for x in tensors]
        self.cpu_rng = torch.get_rng_state()
        self.device_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        self.autocast = torch.is_autocast_enabled(device.type)
        self.autocast_dtype = torch.get_autocast_dtype(device.type)
        self.cache_enabled = torch.is_autocast_cache_enabled()
        self.records, self.values, self.storage_keys = [], [], []
        self.selected, self.mlp_depth = None, 0
        self.action, self.replays, self.backward_started = None, 0, False
        self.invalid_alias = False
        self.nonfinite_codec = False
        self.error_numerator = self.state_squared_norm = 0.0
        self.measure_error = False
        self.value_versions = []

    def pack(self, tensor):
        index = len(self.records)
        self.records.append(_metadata(tensor))
        key = _storage(tensor)
        if self.selected is not None and key == self.storage_keys[self.selected]:
            self.invalid_alias = True
        eligible = (self.mlp_depth > 0 and self.selected is None
                    and tensor.dtype == torch.bfloat16 and tensor.grad_fn is not None
                    and tensor.numel() * tensor.element_size() >= self.minimum_bytes
                    and tensor.is_contiguous() and key not in self.storage_keys
                    and key not in self.checkpoint_storage)
        self.storage_keys.append(key)
        if self.measure_error:
            self.state_squared_norm += tensor.detach().float().square().sum().item()
        if eligible:
            self.selected = index
        value = None if self.representation == 'R' else tensor.detach()
        if eligible and self.representation == 'L':
            value = tensor.detach().to(torch.float8_e4m3fn)
            self.nonfinite_codec = not bool(torch.isfinite(value.to(torch.bfloat16)).all())
            if self.measure_error:
                delta = value.to(torch.bfloat16).float() - tensor.detach().float()
                self.error_numerator = delta.square().sum().item()
        self.values.append(value)
        self.value_versions.append(None if value is None else value._version)
        return index

    @property
    def available(self):
        actions = {'R': {'R'}, 'H': {'R', 'H'}, 'L': {'R', 'L'}}[self.representation].copy()
        if self.selected is None or self.invalid_alias or self.nonfinite_codec:
            actions.discard('L')
        elif self.representation == 'H':
            actions.add('L')
        return actions

    def select(self, action):
        if self.backward_started or self.action is not None:
            raise RuntimeError("Recovery must be selected once before backward")
        if action not in self.available:
            raise ValueError("Requested state is unavailable; exact discarded state cannot be restored")
        self.action = action
        if action == 'R':
            self.values = [None] * len(self.records)
        elif action == 'L' and self.representation == 'H':
            original = self.values[self.selected]
            self.values[self.selected] = original.to(torch.float8_e4m3fn)
            self.value_versions[self.selected] = self.values[self.selected]._version
            if not bool(torch.isfinite(self.values[self.selected].to(torch.bfloat16)).all()):
                self.action = 'R'
                self.values = [None] * len(self.records)
                return self.action
            if self.measure_error:
                delta = self.values[self.selected].to(torch.bfloat16).float() - original.float()
                self.error_numerator = delta.square().sum().item()
        return self.action

    def _replay(self):
        if any(x._version != version for x, version in self.input_versions):
            raise RuntimeError('A checkpoint input was modified before replay')
        index = 0

        def save(tensor):
            nonlocal index
            if index >= len(self.records) or _metadata(tensor) != self.records[index]:
                raise RuntimeError("Replay saved-state metadata changed")
            self.values[index] = tensor.detach()
            index += 1
            if index == len(self.records):
                raise _StopReplay()
            return None

        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.set_rng_state(self.cpu_rng)
            if self.device_rng is not None:
                torch.cuda.set_rng_state(self.device_rng, self.device)
            with torch.enable_grad(), torch.autocast(
                self.device.type, enabled=self.autocast, dtype=self.autocast_dtype,
                cache_enabled=self.cache_enabled
            ), saved_tensors_hooks(save, lambda x: x):
                try:
                    with torch.profiler.record_function('recovery_replay'):
                        self.fn(*self.args, **self.kwargs)
                except _StopReplay:
                    pass
        if index != len(self.records):
            raise RuntimeError("Replay did not recover the complete saved-state bundle")
        self.replays += 1

    def unpack(self, index):
        if self.action is None:
            raise RuntimeError("Post-forward recovery selection is missing")
        self.backward_started = True
        if any(x._version != version for x, version in self.input_versions):
            raise RuntimeError('A checkpoint input was modified before backward')
        if self.action != 'R' and self.values[index] is not None:
            if self.values[index]._version != self.value_versions[index]:
                raise RuntimeError('A retained backward state was modified in place')
        if self.values[index] is None:
            if self.action != 'R' or self.replays:
                raise RuntimeError("Saved states cannot be reused across backward passes")
            self._replay()
        value = self.values[index]
        self.values[index] = None
        if self.action == 'L' and index == self.selected:
            value = value.to(torch.bfloat16)
        return value

    def forward(self, args, kwargs):
        with saved_tensors_hooks(self.pack, self.unpack):
            return self.fn(*args, **kwargs)

    @property
    def reconstruction_error(self):
        return self.error_numerator ** 0.5 / (self.state_squared_norm ** 0.5 + 1e-8)


class Recovery:
    """Install explicit decoder boundaries inside any surrounding FSDP wrappers."""

    def __init__(self, model, method='gc'):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        if method not in ('gc', 'hilore'):
            raise ValueError("Only GC and HiLoRe are supported")
        self.method, self.bundles, self.representations = method, {}, {}
        self.measure_error = False
        self.layers = [m for m in model.modules() if isinstance(m, Qwen2DecoderLayer)]
        self.originals, self.handles = [], []
        for index, layer in enumerate(self.layers):
            original = layer.forward
            self.originals.append((layer, original))

            def forward(module, *args, _index=index, _fn=original, **kwargs):
                if not torch.is_grad_enabled() or not module.training:
                    return _fn(*args, **kwargs)
                if self.method == 'gc':
                    with set_checkpoint_early_stop(True):
                        return checkpoint(_fn, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
                hidden = args[0] if args else kwargs['hidden_states']
                def attributed(*a, **kw):
                    with torch.profiler.record_function(f'unit/{_index}'):
                        return _fn(*a, **kw)
                bundle = Bundle(attributed, args, kwargs, self.representations.get(_index, 'R'), hidden.device)
                bundle.measure_error = self.measure_error
                self.bundles[_index] = bundle
                with torch.profiler.record_function(f'unit/{_index}'):
                    return bundle.forward(args, kwargs)

            layer.forward = types.MethodType(forward, layer)

            def enter(module, args, _index=index):
                bundle = self.bundles.get(_index)
                if bundle is not None:
                    bundle.mlp_depth += 1

            def leave(module, args, output, _index=index):
                bundle = self.bundles.get(_index)
                if bundle is not None:
                    bundle.mlp_depth -= 1

            self.handles.extend([layer.mlp.register_forward_pre_hook(enter),
                                 layer.mlp.register_forward_hook(leave, always_call=True)])

    def begin(self, representations=None):
        if representations is not None and (len(representations) != len(self.layers)
                or any(r not in ('H', 'L', 'R') for r in representations)):
            raise ValueError('Every decoder requires one valid capture representation')
        self.bundles = {}
        self.representations = {} if representations is None else dict(enumerate(representations))

    def select(self, actions):
        if len(actions) != len(self.bundles):
            raise ValueError("Every captured region requires a recovery action")
        return [self.bundles[i].select(action) for i, action in enumerate(actions)]

    def close(self):
        for layer, original in self.originals:
            layer.forward = original
        for handle in self.handles:
            handle.remove()
