"""Token-normalized GRPO objective and current-update exposure."""

import torch


def group_advantages(rewards):
    """Normalize eight binary outcome rewards using population deviation."""
    if rewards.ndim != 2 or rewards.shape[1] != 8:
        raise ValueError("Expected [prompts, 8] outcome rewards")
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("Outcome rewards must be binary")
    rewards = rewards.float()
    return (rewards - rewards.mean(-1, keepdim=True)) / (
        rewards.std(-1, correction=0, keepdim=True) + 1e-6
    )


def loss_and_coefficients(current, old, reference, advantages, mask, update_tokens,
                          clip=0.2, beta=1e-3):
    """Return a microbatch contribution and Eq. 4 under full-update normalization.

    K3 is exp(reference-current) - (reference-current) - 1. The clipping
    boundary convention matches torch.minimum and torch.clamp below.
    Distributed callers multiply the loss by world size before an averaged
    gradient reduction; the returned coefficients retain global normalization.
    """
    if update_tokens <= 0:
        raise ValueError("An actor update must contain optimized response tokens")
    if not (current.shape == old.shape == reference.shape == mask.shape):
        raise ValueError("Log probabilities and mask must share a shape")
    current = current.float() if current.dtype != torch.float64 else current
    old, reference = old.detach().to(current), reference.detach().to(current)
    advantages = advantages.detach().to(current)
    mask = mask.detach().to(current)
    if advantages.ndim == current.ndim - 1:
        advantages = advantages.unsqueeze(-1)
    valid = mask.bool()
    # Padded log probabilities cannot introduce NaNs into either term.
    log_ratio = torch.where(valid, current - old, 0)
    log_reference_ratio = torch.where(valid, reference - current, 0)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)
    k3 = log_reference_ratio.exp() - log_reference_ratio - 1
    loss = (mask * (-torch.minimum(ratio * advantages, clipped * advantages)
                    + beta * k3)).sum() / update_tokens
    with torch.no_grad():
        active = ((advantages >= 0) & (ratio <= 1 + clip)) | (
            (advantages < 0) & (ratio >= 1 - clip))
        omega = mask / update_tokens * (
            -active.to(ratio.dtype) * ratio * advantages + beta * (1 - log_reference_ratio.exp()))
    return loss, omega


def exposure(omega, support, influence=None):
    """Sum each structurally reachable response position once (Eq. 7, B.1)."""
    if support.dtype != torch.bool or support.shape != omega.shape:
        raise ValueError("Support must be a boolean mask matching coefficients")
    weights = support.to(torch.float64)
    if influence is not None:
        if influence.shape != omega.shape or not torch.all(influence >= 0):
            raise ValueError("Influence must be nonnegative and match coefficients")
        weights = weights * influence.double()
    return (omega.detach().double().abs() * weights).sum().item()


def gradient_error(reference, candidate, distributed=False):
    """Eq. 56 over matching disjoint shards, after reduction and before clipping."""
    if reference.keys() != candidate.keys():
        raise ValueError("Gradient parameter sets differ")
    numerator = denominator = 0.0
    for key in reference:
        if reference[key].shape != candidate[key].shape:
            raise ValueError('Gradient shard shapes differ')
        ref, cand = reference[key].float(), candidate[key].float()
        if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
            raise ValueError('Gradient measurements must be finite')
        numerator += (cand - ref).square().sum().item()
        denominator += ref.square().sum().item()
    if distributed:
        import torch.distributed as dist
        device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
        totals = torch.tensor([numerator, denominator], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        numerator, denominator = totals.tolist()
    return (numerator / max(denominator, 1e-30)) ** 0.5
