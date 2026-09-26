"""Single-thread compiled two-budget allocation (Appendices B.2–B.3)."""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def _solve(utility, memory, risk, available, nb, nr):
    count = utility.shape[0]
    previous = np.full((nb + 1, nr + 1), -np.inf, dtype=np.float64)
    previous[0, 0] = 0.0
    choices = np.full((count, nb + 1, nr + 1), -1, dtype=np.int8)
    for unit in range(count):
        current = np.full((nb + 1, nr + 1), -np.inf, dtype=np.float64)
        # Strict improvement preserves action priority R, H, L at a state.
        for action in range(3):
            if not available[unit, action]:
                continue
            dm, dr = memory[unit, action], risk[unit, action]
            for m in range(dm, nb + 1):
                for r in range(dr, nr + 1):
                    value = previous[m - dm, r - dr] + utility[unit, action]
                    if value > current[m, r]:
                        current[m, r] = value
                        choices[unit, m, r] = action
        previous = current
    best, bm, br = -np.inf, 0, 0
    # Terminal ties prefer lower risk and then lower memory.
    for r in range(nr + 1):
        for m in range(nb + 1):
            if previous[m, r] > best:
                best, bm, br = previous[m, r], m, r
    result = np.zeros(count, dtype=np.int8)
    for unit in range(count - 1, -1, -1):
        action = choices[unit, bm, br]
        result[unit] = action
        bm -= memory[unit, action]
        br -= risk[unit, action]
    return result, best


def allocate(utility, memory, risks, representations, static_actions,
             memory_residual, risk_budget):
    """Use measured same-plan costs; columns are R, H, L.

    Positive axes have exactly 1024 intervals. Risk zero disables L even
    for a zero-risk unit. No measured costs or utility values are inferred.
    """
    utility = np.asarray(utility, dtype=np.float64)
    memory = np.asarray(memory, dtype=np.float64)
    risks = np.asarray(risks, dtype=np.float64)
    count = len(representations)
    if utility.shape != (count, 3) or memory.shape != utility.shape or risks.shape != (count,):
        raise ValueError("Invalid allocation dimensions")
    if not all(np.isfinite(x).all() for x in (utility, memory, risks)):
        raise ValueError("Allocation inputs must be finite")
    if np.any(memory < 0) or np.any(risks < 0):
        raise ValueError("Costs must be nonnegative")
    if not all(math.isfinite(v) and v >= 0 for v in (memory_residual, risk_budget)):
        raise ValueError("Budgets must be finite and nonnegative")
    if np.any(utility[:, 0] != 0) or np.any(memory[:, 0] != 0):
        raise ValueError("R must have zero conditional utility and marginal memory")
    allowed = np.zeros((count, 3), dtype=np.bool_)
    for i, (representation, actions) in enumerate(zip(representations, static_actions, strict=True)):
        if representation not in 'HLR' or len(representation) != 1 or 'R' not in actions:
            raise ValueError("Invalid retained representation or action set")
        retained = {'H': 'RHL', 'L': 'RL', 'R': 'R'}[representation]
        for a, name in enumerate('RHL'):
            allowed[i, a] = name in retained and name in actions
            if a and utility[i, a] <= 0:
                allowed[i, a] = False
        if risk_budget == 0:
            allowed[i, 2] = False
    nb = 1024 if memory_residual > 0 else 0
    nr = 1024 if risk_budget > 0 else 0
    bm, br = np.zeros_like(memory, dtype=np.int64), np.zeros_like(memory, dtype=np.int64)
    for i in range(count):
        for a in range(3):
            if memory_residual == 0:
                allowed[i, a] &= memory[i, a] == 0
            elif memory[i, a] > memory_residual:
                allowed[i, a] = False
            else:
                bm[i, a] = math.ceil(memory[i, a] / (memory_residual / 1024))
        if risk_budget > 0:
            if risks[i] > risk_budget:
                allowed[i, 2] = False
            else:
                br[i, 2] = math.ceil(risks[i] / (risk_budget / 1024))
    actions, value = _solve(utility, bm, br, allowed, nb, nr)
    return ['RHL'[int(a)] for a in actions], float(value)


def warmup():
    """Compile the CPU solver before the actor-update timing window."""
    _solve(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64),
           np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.bool_), 0, 0)
