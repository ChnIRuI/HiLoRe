"""Four-rank FSDP actor updates over fixed GRPO replay tensors."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from functools import partial
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from .data import load_update, microbatches
from .splits import SplitManifest
from .objective import loss_and_coefficients
from .recovery import Recovery
from .runtime import Scheduler


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs/qwen25_3b_deepmath.json'
MODEL_CONFIG = {'model_type': 'qwen2', 'num_hidden_layers': 36, 'hidden_size': 2048,
                'intermediate_size': 11008, 'num_attention_heads': 16, 'num_key_value_heads': 2,
                'vocab_size': 151936, 'eos_token_id': 151645, 'attention_dropout': 0.,
                'hidden_act': 'silu', 'tie_word_embeddings': True, 'use_sliding_window': False,
                'rope_theta': 1000000., 'rms_norm_eps': 1e-6}


def validate_model_configuration(value):
    for key, expected in MODEL_CONFIG.items():
        if value.get(key) != expected:
            raise ValueError(f'Expected Qwen2.5-3B-Instruct configuration: {key} must be {expected}')


def fingerprint(config, model_identity=None):
    execution = {k: v for k, v in config.items() if k not in ('model', 'method', 'risk_budget', 'method_seed')}
    execution['software'] = {name: importlib.metadata.version(name) for name in
                             ('torch', 'transformers', 'peft', 'accelerate', 'numpy', 'numba', 'tokenizers', 'llvmlite')}
    execution['cuda'] = torch.version.cuda
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    execution['implementation_sha256'] = code.hexdigest()
    execution['model_identity'] = model_identity
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        execution['device'] = {'name': properties.name, 'total_memory': properties.total_memory,
                               'capability': [properties.major, properties.minor]}
    return hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()


def model_identity(model_path):
    """Bind a profile to local weight bytes or an immutable Hub commit."""
    path = Path(model_path)
    if path.is_dir():
        files = sorted(set(path.glob('*.safetensors')) | set(path.glob('pytorch_model*.bin')))
        if not files or not (path / 'config.json').is_file():
            raise ValueError('A local model needs config.json and complete weight files')
        validate_model_configuration(json.loads((path / 'config.json').read_text()))
        digest = hashlib.sha256()
        for file in [path / 'config.json'] + files:
            digest.update(file.name.encode())
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(chunk)
        return {'weight_sha256': digest.hexdigest()}
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    validate_model_configuration(model_config.to_dict())
    revision = model_config._commit_hash
    if not revision:
        raise ValueError('Cannot establish an immutable model revision')
    return {'hub_id': model_path, 'revision': revision}


def build_model(model_path, config, device, identity=None):
    """Retain FP32 persistent parameters; FSDP handles BF16 compute casts."""
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False, revision=(identity or {}).get('revision'))
    validate_model_configuration(hf_config.to_dict())
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, config['persistent_precision']),
                    attn_implementation='sdpa', trust_remote_code=False,
                    revision=(identity or {}).get('revision'), config=hf_config)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=config['lora_rank'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
        target_modules=config['lora_targets']))
    model.enable_input_require_grads()
    model.gradient_checkpointing_disable()
    recovery = Recovery(model, config['method'])
    model = FSDP(model, auto_wrap_policy=partial(transformer_auto_wrap_policy,
                    transformer_layer_cls={Qwen2DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=config['use_orig_params'],
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, config['compute_precision']),
                    reduce_dtype=getattr(torch, config['reduction_precision']),
                    buffer_dtype=getattr(torch, config['buffer_precision'])),
        device_id=device, forward_prefetch=config['forward_prefetch'],
        backward_prefetch=BackwardPrefetch[config['backward_prefetch']],
        limit_all_gathers=config['limit_all_gathers'],
        sync_module_states=True)
    return model.train(), recovery


def setup_distributed(config):
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']
    torch.use_deterministic_algorithms(config['strict_determinism'])
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    return torch.device('cuda', rank)


def make_optimizer(model, config):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'], betas=tuple(config['betas']), eps=config['adam_epsilon'],
        weight_decay=config['weight_decay'])


def gradient_snapshot(model):
    """Copy disjoint FP32 FSDP gradient shards without retaining device storage."""
    return {name: (p.grad.detach().float().cpu().clone() if p.grad is not None
                   else torch.zeros(p.shape, dtype=torch.float32))
            for name, p in model.named_parameters() if p.requires_grad}


def backward_batch(model, recovery, scheduler, batch, total_tokens, config, device, observer=None):
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    shape = tuple(batch['input_ids'].shape)
    representations = scheduler.before_forward(shape, len(recovery.layers)) if scheduler else None
    recovery.begin(representations)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        positions = batch['attention_mask'].long().cumsum(-1).sub(1).clamp_min(0)
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       position_ids=positions, use_cache=False).logits[:, :-1].float()
        current = (logits / config['rollout_temperature']).log_softmax(-1).gather(
            -1, batch['input_ids'][:, 1:, None]).squeeze(-1)
        loss, omega = loss_and_coefficients(current, batch['old_log_probs'],
            batch['reference_log_probs'], batch['advantages'], batch['response_mask'],
            total_tokens, config['grpo_clip'], config['kl_coefficient'])
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite actor loss')
    if scheduler:
        scheduler.after_forward(recovery, omega, batch['response_mask'],
                                int(batch['response_mask'].sum()), total_tokens)
    if observer:
        observer(recovery, omega, batch)
    (loss * dist.get_world_size()).backward()


def actor_update(model, optimizer, recovery, scheduler, data, config, device,
                 collect_gradients=False, observer=None):
    """Reduce each microbatch, accumulate in FP32, then take one AdamW step."""
    rank, world = dist.get_rank(), dist.get_world_size()
    total_tokens = int(data['response_mask'].sum())
    optimizer.zero_grad(set_to_none=True)
    for batch in microbatches(data, rank, world, config['microbatch_token_target_per_rank']):
        backward_batch(model, recovery, scheduler, batch, total_tokens, config, device, observer)
    gradients = gradient_snapshot(model) if collect_gradients else None
    norm = model.clip_grad_norm_(config['gradient_clip'])
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite accumulated actor gradient')
    optimizer.step()
    recovery.bundles.clear()
    if collect_gradients:
        return {'complete_update': True, 'after_reduction_before_clipping': True,
                'gradients': gradients, 'gradient_norm': float(norm)}
    return float(norm)


def validate_configuration(config):
    """Reject silent changes to the supported Table 6 execution configuration."""
    fixed = {'train_prompts_per_update': 512, 'responses_per_prompt': 8,
        'prompt_limit': 1024, 'response_limit': 2048, 'microbatch_token_target_per_rank': 4096,
        'remove_padding': False, 'actor_epochs': 1, 'learning_rate': 2e-6, 'optimizer': 'AdamW',
        'betas': [.9, .999], 'adam_epsilon': 1e-8, 'weight_decay': 0., 'gradient_clip': 1.,
        'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': 0.,
        'lora_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'grpo_clip': .2, 'kl_coefficient': .001, 'advantage_epsilon': 1e-6,
        'rollout_temperature': .6, 'rollout_top_p': 1., 'rollout_top_k': 0,
        'world_size': 4, 'persistent_precision': 'float32', 'compute_precision': 'bfloat16',
        'reduction_precision': 'float32', 'buffer_precision': 'float32', 'use_orig_params': True,
        'forward_prefetch': False, 'backward_prefetch': 'BACKWARD_PRE', 'limit_all_gathers': True,
        'checkpoint_reentrant': False, 'preserve_rng_state': True, 'checkpoint_early_stop': True,
        'tf32': False, 'strict_determinism': False, 'calibration_microbatches': 16,
        'memory_intervals': 1024, 'risk_intervals': 1024, 'gradient_error_threshold': .015,
        'minimum_compressed_tensor_bytes': 32 << 20, 'maximum_compressed_tensors_per_layer': 1}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported paper configuration: {key} must be {value}')
    if not isinstance(config['seed'], int) or not isinstance(config['updates'], int) or config['updates'] <= 0:
        raise ValueError('Seed and a positive update count must be integers')


def replay_digests(files):
    from .prepare_data import file_digest
    return [file_digest(path) for path in files]


def checkpoint_context(config, identity, scheduler, digests):
    profile = scheduler.profile if scheduler else None
    return {'configuration_fingerprint': fingerprint(config, identity), 'method': config['method'],
            'world_size': dist.get_world_size(), 'replay_sha256': digests,
            'profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()}


def save_checkpoint(directory, step, model, optimizer, recovery, context):
    """Save exact local adapter/optimizer shards and RNG at an update boundary."""
    from .profiling import Snapshot
    error = [None]
    if dist.get_rank() == 0:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    rank = dist.get_rank()
    snapshot = Snapshot(model, optimizer, recovery)
    temporary = directory / f'rank-{rank}.tmp'
    torch.save({'context': context, 'step': step, 'rank': rank, 'state': snapshot.state_dict()}, temporary)
    temporary.replace(directory / f'rank-{rank}.pt')
    dist.barrier()
    if rank == 0:
        (directory / 'complete.json').write_text(json.dumps({'step': step, 'world_size': dist.get_world_size()}))
    dist.barrier()


def resume_checkpoint(directory, model, optimizer, recovery, context):
    from .profiling import Snapshot
    complete = json.loads((directory / 'complete.json').read_text())
    state = torch.load(directory / f'rank-{dist.get_rank()}.pt', map_location='cpu', weights_only=True)
    if (state['context'] != context or state['rank'] != dist.get_rank() or state['step'] != complete['step']
            or complete['world_size'] != dist.get_world_size()):
        raise ValueError('Checkpoint model, replay, method, profile or rank identity differs')
    Snapshot(model, optimizer, recovery).load_state_dict(state['state'])
    return state['step']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--model', default=None)
    parser.add_argument('--method', choices=['gc', 'hilore'], required=True)
    parser.add_argument('--updates', type=Path, required=True, help='Directory of update_*.pt replay tensors')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--risk-budget', type=float)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-every', type=int, default=20)
    parser.add_argument('--development-profile', action='store_true',
                        help='Permit a measured development profile before terminal quality selection')
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        parser.error('Checkpoint interval must be positive')
    config = json.loads(args.config.read_text())
    validate_configuration(config)
    config['method'] = args.method
    if args.risk_budget is not None:
        config['risk_budget'] = args.risk_budget
    if args.method == 'hilore' and args.profile is None:
        parser.error('HiLoRe requires a measured, validated profile; no synthetic profile is supplied')
    if int(os.environ.get('WORLD_SIZE', '1')) != 4:
        parser.error('The paper configuration requires four FSDP ranks')
    manifest = SplitManifest(args.split_manifest)
    config['split_manifest_sha256'] = manifest.sha256
    files = sorted(args.updates.glob('update_*.pt'))
    if len(files) != config['updates']:
        parser.error('Provide exactly the configured number of complete replay updates')
    # Inspect every update before allocating the model or applying any gradients.
    for path in files:
        load_update(path, config, manifest)
    if args.model:
        config['model'] = args.model
    identity = model_identity(config['model'])
    device = setup_distributed(config)
    scheduler = Scheduler(args.profile, fingerprint(config, identity), config['memory_multiplier'],
                          config['risk_budget'], allow_development=args.development_profile) if args.method == 'hilore' else None
    error = [None]
    if dist.get_rank() == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        dist.destroy_process_group()
        raise RuntimeError(error[0])
    model, recovery = build_model(config['model'], config, device, identity)
    optimizer = make_optimizer(model, config)
    context = checkpoint_context(config, identity, scheduler, replay_digests(files))
    start = resume_checkpoint(args.resume, model, optimizer, recovery, context) if args.resume else 0
    if not 0 <= start <= len(files):
        raise ValueError('Checkpoint update index is outside the replay sequence')
    try:
        for step, path in enumerate(files[start:], start=start):
            data = load_update(path, config, manifest)
            norm = actor_update(model, optimizer, recovery, scheduler, data, config, device)
            if dist.get_rank() == 0:
                print(json.dumps({'update': step + 1, 'gradient_norm': norm}), flush=True)
            if (step + 1) % args.checkpoint_every == 0 or step + 1 == len(files):
                save_checkpoint(args.output / f'checkpoint-{step + 1:04d}', step + 1,
                                model, optimizer, recovery, context)
    finally:
        recovery.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
