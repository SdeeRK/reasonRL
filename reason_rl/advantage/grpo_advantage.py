"""
Advantage computation for GRPO training.

Transforms RewardBatch → AdvantageBatch by normalizing rewards within groups.
"""

import numpy as np

from ..core.config import AdvantageConfig
from ..core.types import AdvantageSample, RewardBatch


class GRPOAdvantage:
    """
    Compute advantages from rewards with configurable normalization.

    Implements AdvantageComputerProtocol with three modes:
    - "raw": Return original reward as advantage (no normalization)
    - "grpo": Standard GRPO (subtract mean, divide by std per group)
    - "grpo_no_std": Only subtract mean per group

    The normalization is done within each group (per-prompt), comparing
    responses to the same prompt against each other.
    """

    def __init__(self, config: AdvantageConfig):
        self.config = config

    def __call__(self, reward_batch: RewardBatch) -> list[AdvantageSample]:
        """
        Compute advantages for all rollouts in batch.

        Args:
            reward_batch: RewardBatch containing groups and their rewards.

        Returns:
            List of AdvantageSample with computed advantages.
        """
        advantage_samples: list[AdvantageSample] = []
        for group_rollout, group_reward in zip(reward_batch.groups, reward_batch.rewards):
            group_advantage = self._compute_group_advantage(group_reward)
            for rollout, advantage in zip(group_rollout.items, group_advantage):
                advantage_samples.append(
                    AdvantageSample(
                        prompt_ids=group_rollout.prompt_ids,
                        response_ids=rollout.response_ids,
                        old_log_probs=rollout.old_log_probs,
                        advantage=advantage,
                    )
                )

        return advantage_samples

    def _compute_group_advantage(self, group_reward: list[float]) -> list[float]:
        if self.config.mode == "raw":
            return group_reward

        rewards = np.array(group_reward)
        mean = np.mean(rewards)

        if self.config.mode == "grpo_no_std":
            return (rewards - mean).tolist()

        if self.config.mode == "grpo":
            std = np.std(rewards)
            return ((rewards - mean) / (std + self.config.advantage_eps)).tolist()

        raise ValueError(f"Invalid advantage mode: {self.config.mode}")
