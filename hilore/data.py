"""Dense, fixed-workload GRPO replay input and deterministic microbatching."""

from pathlib import Path
import torch
from .objective import group_advantages
from .splits import validate_replay_provenance


REQUIRED = {'input_ids', 'attention_mask', 'response_mask', 'old_log_probs',
            'reference_log_probs', 'rewards', 'prompt_ids'}


def load_update(path, config, manifest=None):
    """Read replay tensors and primitive provenance fields. Token fields align with ids[:, 1:]."""
    if manifest is None:
        raise ValueError('A validated split manifest is required for actor replay')
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or not REQUIRED <= data.keys():
        raise ValueError('Missing replay tensor fields')
    if data.get('log_prob_temperature') != config.get('rollout_temperature', .6):
        raise ValueError('Replay log-probability temperature must match the actor')
    if any(not isinstance(data[k], torch.Tensor) for k in REQUIRED):
        raise ValueError('Replay fields must be tensors')
    ids, mask = data['input_ids'], data['attention_mask']
    if ids.ndim != 2 or ids.dtype != torch.long or torch.any(ids < 0):
        raise ValueError('input_ids must be a nonnegative int64 matrix')
    count, length = ids.shape
    if length < 2:
        raise ValueError('Each sequence needs a prompt and a response')
    if mask.shape != ids.shape or count != config['train_prompts_per_update'] * 8:
        raise ValueError('Replay does not match the complete actor-update workload')
    if length > config['prompt_limit'] + config['response_limit']:
        raise ValueError('Replay sequence exceeds the configured limit')
    for key in ['response_mask', 'old_log_probs', 'reference_log_probs']:
        if data[key].shape != (count, length - 1):
            raise ValueError(f'Invalid shape: {key}')
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all((data['response_mask'] == 0) | (data['response_mask'] == 1)):
        raise ValueError('Masks must be binary')
    if torch.any(data['response_mask'] > mask[:, 1:]):
        raise ValueError('Response mask includes padding')
    valid = mask.bool()
    response = torch.zeros_like(valid)
    response[:, 1:] = data['response_mask'].bool()
    prompt = valid & ~response
    # Each row is padding, prompt, response, padding, with contiguous spans.
    for span in (valid, prompt, response):
        starts = span[:, :1].sum(-1) + (span[:, 1:] & ~span[:, :-1]).sum(-1)
        if torch.any(starts != 1):
            raise ValueError('Prompt and response must be nonempty contiguous spans')
    if torch.any(response[:, :-1] & prompt[:, 1:]):
        raise ValueError('Response tokens must follow the prompt')
    if torch.any(response[:, 1:] & ~valid[:, :-1]):
        raise ValueError('A response target must have a valid preceding token')
    for key in ('old_log_probs', 'reference_log_probs'):
        values = data[key]
        if not values.is_floating_point():
            raise ValueError(f'{key} must be floating point')
        selected = values[data['response_mask'].bool()]
        if not torch.isfinite(selected).all() or torch.any(selected > 1e-5):
            raise ValueError(f'{key} contains invalid sampled-token log probabilities')
    if torch.any(data['response_mask'].sum(-1) > config['response_limit']):
        raise ValueError('Response limit exceeded')
    prompts = mask.sum(-1) - data['response_mask'].sum(-1)
    if torch.any(prompts > config['prompt_limit']):
        raise ValueError('Prompt limit exceeded')
    prompt_ids = data['prompt_ids']
    if prompt_ids.shape != (count,) or data['rewards'].shape != (count,):
        raise ValueError('Prompt IDs and rewards must have one entry per response')
    if prompt_ids.dtype != torch.long:
        raise ValueError('prompt_ids must be int64')
    validate_replay_provenance(data, prompt_ids.tolist(), manifest)
    unique, counts = torch.unique(prompt_ids, return_counts=True)
    if len(unique) != config['train_prompts_per_update'] or not torch.all(counts == 8):
        raise ValueError('Every prompt must retain all eight responses')
    advantage = torch.empty(count, dtype=torch.float32)
    for prompt in unique:
        selected = prompt_ids == prompt
        rows = torch.where(selected)[0]
        prompt_tokens = [ids[i][valid[i] & ~response[i]] for i in rows]
        if any(not torch.equal(prompt_tokens[0], tokens) for tokens in prompt_tokens[1:]):
            raise ValueError('Responses in a group must share identical prompt tokens')
        advantage[selected] = group_advantages(data['rewards'][selected].reshape(1, 8))[0]
    data['advantages'] = advantage
    if data['response_mask'].sum() <= 0:
        raise ValueError('No optimized response tokens')
    return data


def microbatches(data, rank, world_size, token_target=4096):
    """Preserve every sequence; dense cost includes padding within each batch.

    Equal-size rank shards use the same conservative sequence-count partition
    to keep FSDP collectives aligned. No token selection or truncation occurs.
    """
    count, length = data['input_ids'].shape
    if world_size <= 0 or not 0 <= rank < world_size or token_target <= 0:
        raise ValueError('Invalid rank or microbatch target')
    if count % world_size:
        raise ValueError('Update size must divide evenly across ranks')
    per_rank = count // world_size
    batch_size = max(1, token_target // length)
    start, end = rank * per_rank, (rank + 1) * per_rank
    for offset in range(start, end, batch_size):
        stop = min(offset + batch_size, end)
        yield {k: v[offset:stop] for k, v in data.items() if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == count}
