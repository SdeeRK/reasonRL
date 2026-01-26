import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict

from torch.utils.data import DataLoader

from ..core.config import RLConfig
from ..core.types import StepMetrics

logger = logging.getLogger(__name__)

# Lazy import wandb to avoid import errors if not installed
try:
    import wandb
except ImportError:
    wandb = None


class BaseTrainer(ABC):
    """
    Abstract base class for RL trainers.

    Provides common functionality:
    - WandB initialization and logging
    - Checkpoint saving with artifact upload
    - Dataloader iteration helper
    """

    config: RLConfig
    train_dataloader: DataLoader
    global_step: int
    _train_iter: Iterator | None

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging if enabled."""
        wandb_config = self.config.wandb_config

        if not wandb_config.enabled:
            logger.info("WandB logging disabled")
            return

        if wandb is None:
            logger.warning(
                "wandb not installed, logging disabled. Run: pip install wandb"
            )
            self.config.wandb_config.enabled = False
            return

        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=wandb_config.run_name,
            tags=wandb_config.tags,
            config={
                "model": asdict(self.config.model_config),
                "generation": asdict(self.config.generation_config),
                "train": asdict(self.config.train_config),
                "loss": asdict(self.config.loss_config),
            },
        )
        logger.info(f"WandB initialized: {wandb.run.name}")

    def _get_next_batch(self):
        """Get next batch from dataloader, cycling if exhausted."""
        if self._train_iter is None:
            self._train_iter = iter(self.train_dataloader)

        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)

        return batch

    def _log_metrics(
        self,
        metrics: StepMetrics,
        step: int,
        epoch: int,
        rollout_stats: dict[str, float] | None = None,
    ) -> None:
        """Log training metrics to console and WandB."""
        wandb_config = self.config.wandb_config

        # Prepare log dict
        log_dict = {
            "train/loss": metrics.loss,
            "train/learning_rate": metrics.learning_rate,
            "train/step": step,
            "train/epoch": epoch,
        }

        if metrics.entropy is not None:
            log_dict["train/entropy"] = metrics.entropy

        if metrics.gradient_norm is not None:
            log_dict["train/gradient_norm"] = metrics.gradient_norm

        # Add custom stats from loss function
        for key, value in metrics.stats.items():
            log_dict[f"train/{key}"] = value

        # Add rollout stats (only on first epoch to avoid duplication)
        if epoch == 0 and rollout_stats:
            log_dict.update(rollout_stats)

        # Log to WandB
        if (
            wandb_config.enabled
            and wandb is not None
            and step % wandb_config.log_interval == 0
        ):
            wandb.log(log_dict, step=step)

        # Log to console
        lr_str = f"{metrics.learning_rate:.2e}" if metrics.learning_rate else "N/A"
        entropy_str = f" | Entropy: {metrics.entropy:.4f}" if metrics.entropy else ""

        logger.info(
            f"Step {step + 1}/{self.config.train_config.n_grpo_steps} "
            f"Epoch {epoch + 1}/{self.config.train_config.epochs_per_rollout_batch} | "
            f"Loss: {metrics.loss:.4f} | LR: {lr_str}{entropy_str}"
        )

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log evaluation metrics to WandB."""
        if self.config.wandb_config.enabled and wandb is not None:
            wandb.log(metrics, step=step)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        raise NotImplementedError("Subclass must implement save_checkpoint")

    @abstractmethod
    def train(self) -> None:
        """Main training loop."""
        ...

    def save_eval_results(self, results: list[dict], step: int) -> None:
        """
        Save evaluation results to JSONL file.

        Args:
            results: List of result dictionaries containing prompt, response, ground_truth, reward.
            step: Current training step.
        """
        import json
        import os

        results_dir = "eval_results"
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"step_{step}.jsonl")

        with open(output_file, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Saved evaluation results to {output_file}")

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation."""
        ...

    def _get_checkpoint_path(self, name: str) -> str:
        """
        Get absolute path for a checkpoint, creating the directory if needed.

        Args:
            name: Name of the checkpoint (e.g., "checkpoint-step-10").

        Returns:
            Absolute path to the checkpoint directory.
        """
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, name)
