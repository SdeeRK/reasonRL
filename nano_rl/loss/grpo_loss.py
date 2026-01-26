import torch
import torch.nn.functional as F

from ..core.config import LossConfig
from ..core.types import StepMetrics


class GRPOLoss:
    """
    GRPO (Group Relative Policy Optimization) loss with PPO-style clipping.
    """

    def __init__(self, config: LossConfig):
        self.config = config

    def __call__(
        self,
        model_logits: torch.Tensor,
        old_log_probs: torch.Tensor,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        return_token_entropy: bool,
    ) -> StepMetrics:
        """
        Compute GRPO loss with PPO-style clipping.

        Args:
            model_logits: (batch_size, seq_len, vocab_size) - raw logits from model
            old_log_probs: (batch_size, seq_len) - log probs from rollout policy
            labels: (batch_size, seq_len) - target token ids
            response_mask: (batch_size, seq_len) - 1 for response tokens, 0 for prompt
            advantages: (batch_size,) or (batch_size, 1) - advantage values per sequence
            return_token_entropy: whether to compute and return entropy

        Returns:
            StepMetrics with loss, entropy, and clipping statistics
        """
        # ==================== 1. Compute policy log probs ====================
        # log_softmax first, then gather the target token's log prob
        log_probs = F.log_softmax(model_logits, dim=-1)  # (batch, seq, vocab)
        policy_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq)

        # ==================== 2. Compute ratio ====================
        log_ratio = policy_log_probs - old_log_probs
        ratio = log_ratio.exp()

        # Clamp ratio for numerical stability
        ratio_clipped = torch.clamp(
            ratio,
            1 - self.config.clip_range_left,
            1 + self.config.clip_range_right,
        )

        # ==================== 3. Compute surrogate loss ====================
        # Expand advantages to match token dimension if needed
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)  # (batch, 1)

        surrogate1 = ratio * advantages
        surrogate2 = ratio_clipped * advantages

        # PPO-clip objective: min(surrogate1, surrogate2)
        # We want to maximize this, so loss = -min(...)
        token_loss = -torch.minimum(surrogate1, surrogate2)

        # Apply response mask - only compute loss on response tokens
        token_loss = token_loss * response_mask

        # ==================== 4. Aggregate loss ====================
        if self.config.loss_level == "token":
            # Average over all response tokens
            loss = token_loss.sum() / response_mask.sum().clamp(min=1)
        elif self.config.loss_level == "sequence":
            # First average per sequence, then average across batch
            seq_loss = token_loss.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
            loss = seq_loss.mean()
        else:
            raise ValueError(f"Invalid loss level: {self.config.loss_level}")

        # ==================== 5. Compute entropy (optional) ====================
        entropy = None
        if return_token_entropy:
            entropy = self._compute_entropy(model_logits, log_probs, response_mask)

        # ==================== 6. Compute clipping statistics ====================
        stats = self._compute_clip_stats(ratio, surrogate1, surrogate2, response_mask)

        return StepMetrics(
            loss=loss,
            entropy=entropy,
            stats=stats,
        )

    def _compute_entropy(
        self,
        model_logits: torch.Tensor,
        log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> float:
        """Compute average token entropy over response tokens."""
        with torch.inference_mode():
            # Entropy = -sum(p * log(p)) over vocab dimension
            probs = F.softmax(model_logits, dim=-1)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq)
            token_entropy = token_entropy * response_mask

            if self.config.loss_level == "token":
                entropy = token_entropy.sum() / response_mask.sum().clamp(min=1)
            else:
                seq_entropy = token_entropy.sum(dim=1) / response_mask.sum(dim=1).clamp(
                    min=1
                )
                entropy = seq_entropy.mean()

            return entropy.item()

    def _compute_clip_stats(
        self,
        ratio: torch.Tensor,
        surrogate1: torch.Tensor,
        surrogate2: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Compute clipping statistics for logging."""
        with torch.inference_mode():
            # Only consider response tokens for statistics
            response_surrogate1 = surrogate1 * response_mask
            response_surrogate2 = surrogate2 * response_mask

            # Token is clipped when surrogate2 != surrogate1 (i.e., clipping was applied)
            is_clipped = (response_surrogate2.abs() < response_surrogate1.abs()) & (
                response_mask > 0
            )

            # Upper clipping: ratio > 1 + eps
            clip_upper = is_clipped & (ratio > 1 + self.config.clip_range_right)
            # Lower clipping: ratio < 1 - eps
            clip_lower = is_clipped & (ratio < 1 - self.config.clip_range_left)

            num_response_tokens = response_mask.sum().clamp(min=1)
            frac_clip_total = is_clipped.float().sum() / num_response_tokens
            frac_clip_upper = clip_upper.float().sum() / num_response_tokens
            frac_clip_lower = clip_lower.float().sum() / num_response_tokens

            # Mean ratio (only for response tokens)
            mean_ratio = (ratio * response_mask).sum() / num_response_tokens

        return {
            "clip_frac_total": frac_clip_total.item(),
            "clip_frac_upper": frac_clip_upper.item(),
            "clip_frac_lower": frac_clip_lower.item(),
            "mean_ratio": mean_ratio.item(),
        }
