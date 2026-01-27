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
        format_reward: float = 1.0,
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
        all_reward_details: list[list[dict[str, float]]] = []

        for group in rollout_batch.groups:
            group_rewards = []
            group_reward_details = []
            ground_truth = group.sample.ground_truth or ""

            for rollout in group.items:
                metrics = self.compute(rollout.response_text, ground_truth)
                reward = metrics.get("reward", 0.0)
                group_rewards.append(reward)
                group_reward_details.append(metrics)

            all_rewards.append(group_rewards)
            all_reward_details.append(group_reward_details)

        return RewardBatch(groups=rollout_batch.groups, rewards=all_rewards, reward_details=all_reward_details)

    def compute(self, response: str, ground_truth: str) -> dict[str, float]:
        """Compute reward for a single response.
        return dict containing reward details.
        Total reward must be in 'reward' key.
        detail reward is {
            'reward': reward,
            'format_reward': format_reward,
            'answer_reward': answer_reward,
        }
        """
        return self._compute_reward_with_metrics(response, ground_truth)

    def _compute_reward_with_metrics(self, response: str, ground_truth: str) -> dict[str, float]:
        """Compute reward and return metrics dict."""
        total_reward = 0.0
        format_score = 0.0
        answer_score = 0.0

        if self._check_format(response):
            format_score = self.format_reward

        if self._check_answer(response, ground_truth):
            answer_score = self.answer_reward

        if format_score and answer_score:
            total_reward = answer_score

        metrics = {
            "reward": total_reward,
            "format_reward": format_score,
            "answer_reward": answer_score,
        }

        return metrics

    def _check_format(self, response: str) -> bool:
        """
        Check if response has properly formatted tags.

        Valid format: <think>...</think><answer>...</answer>
        """

    def _check_format(self, response: str) -> bool:
        """
        Check if response has properly formatted tags.

        Valid format: ...</think><answer>...</answer>
        (Opening <think> might be in the prompt, so it's optional in response)
        Strict check:
        - </think> appears exactly once
        - <answer> appears exactly once
        - </answer> appears exactly once
        - Order: </think> -> <answer> -> </answer>
        """
        # 1. Check counts
        if response.count("</think>") != 1:
            return False
        if response.count("<answer>") != 1:
            return False
        if response.count("</answer>") != 1:
            return False

        # 2. Check order
        think_end = response.find("</think>")
        answer_start = response.find("<answer>")
        answer_end = response.find("</answer>")

        # </think> must be before <answer>
        if think_end >= answer_start:
            return False

        # <answer> must be before </answer>
        if answer_start >= answer_end:
            return False

        return True

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
