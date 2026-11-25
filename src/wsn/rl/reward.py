"""Per-attempt reward for the Q-Learning router."""

from __future__ import annotations

from wsn.config import RewardWeights
from wsn.models import HopOutcome
from wsn.rl.state import Candidate


def attempt_reward(
    weights: RewardWeights,
    chosen: Candidate,
    outcome: HopOutcome,
    reference_energy_j: float,
    communication_range_m: float,
) -> float:
    """Base terms once per attempted hop, then the terminal bonus or penalty.

    ``r = -hop - energy*(charged/E_ref) - util*u(v) - depletion*(1-e(v))
         + progress*clip(progress/range) - exhaustion*newly_exhausted
         + bonus if delivered, - penalty if the packet failed on this attempt``
    Neighbor features are the values observed when the action was chosen.
    """
    progress = min(max(chosen.progress_m / communication_range_m, -1.0), 1.0)
    reward = (
        -weights.hop_penalty
        - weights.energy_cost * (outcome.charged_energy_j / reference_energy_j)
        - weights.utilization * chosen.utilization
        - weights.depletion * (1.0 - chosen.energy_fraction)
        + weights.progress * progress
        - weights.exhaustion_penalty * outcome.newly_exhausted_nodes
    )
    if outcome.delivered:
        return reward + weights.delivered_bonus
    if outcome.terminal_outcome is not None:
        return reward - weights.failure_penalty
    return reward
