import re

from math_verify import parse, verify

from ..core.types import RewardBatch, RolloutBatch


class MathReward:
    """
    Reward computer for R1-Zero style math training.

    Implements RewardComputerProtocol with:
    - Format checking (think/answer tags)
    - Math answer verification using math-verify library
    """

    def __init__(
        self,
        format_reward: float = 0.5,
        answer_reward: float = 1.0,
    ):
        """
        Args:
            format_reward: Reward for correct format (tags properly closed).
            answer_reward: Reward for correct answer.
        """
        self.format_reward = format_reward
        self.answer_reward = answer_reward

        # Regex patterns for R1-Zero format
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def __call__(self, rollout_batch: RolloutBatch) -> RewardBatch:
        """Compute rewards for all rollouts in batch."""
        all_rewards: list[list[float]] = []

        for group in rollout_batch.groups:
            group_rewards = []
            ground_truth = group.sample.ground_truth or ""

            for rollout in group.items:
                reward = self.compute(rollout.response_text, ground_truth)
                group_rewards.append(reward)

            all_rewards.append(group_rewards)

        return RewardBatch(groups=rollout_batch.groups, rewards=all_rewards)

    def compute(self, response: str, ground_truth: str) -> float:
        """Compute reward for a single response."""
        total_reward = 0.0

        # 1. Format check (no penalty, just no reward if wrong)
        if self._check_format(response):
            total_reward += self.format_reward

        # 2. Answer check using math-verify
        if self._check_answer(response, ground_truth):
            total_reward += self.answer_reward

        return total_reward

    def _check_format(self, response: str) -> bool:
        """
        Check if response has properly formatted tags.

        Valid format: <think>...</think><answer>...</answer>
        """
        # Check for opening and closing tags
        has_think_open = "<think>" in response
        has_think_close = "</think>" in response
        has_answer_open = "<answer>" in response
        has_answer_close = "</answer>" in response

        # All tags must be present
        if not all(
            [has_think_open, has_think_close, has_answer_open, has_answer_close]
        ):
            return False

        # Check order: think should come before answer
        think_match = self.think_pattern.search(response)
        answer_match = self.answer_pattern.search(response)

        if not think_match or not answer_match:
            return False

        # Think should end before answer starts
        return think_match.end() <= answer_match.start()

    def _check_answer(self, response: str, ground_truth: str) -> bool:
        """
        Check if the extracted answer matches ground truth using math-verify.

        Extracts answer directly from <answer>...</answer> tag content.
        """
        if not ground_truth:
            return False

        # Extract answer from <answer> tag
        answer_match = self.answer_pattern.search(response)
        if not answer_match:
            return False

        pred_answer = answer_match.group(1).strip()

        try:
            # Use math-verify for comparison
            pred_parsed = parse(pred_answer)
            gt_parsed = parse(ground_truth)
            return verify(pred_parsed, gt_parsed)
        except Exception:
            # Fallback to string comparison if parsing fails
            return pred_answer.strip() == ground_truth.strip()
