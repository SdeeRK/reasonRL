import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from ..core.config import RLConfig
from ..core.types import Sample
from ..logger import get_logger

logger = get_logger(__name__)


try:
    import wandb
except ImportError:
    wandb = None


class BaseTrainer(ABC):
    """
    Abstract base class for Reinforcement Learning trainers.

    This class defines the interface for all trainer implementations and provides
    common utilities for logging, checkpointing, and data iteration.
    """

    config: RLConfig
    train_dataloader: DataLoader
    global_step: int
    _train_iter: Iterator | None

    def __init__(self, config: RLConfig):
        logger.info(f"Training Config:\n{json.dumps(asdict(config), indent=4, default=str)}")
        self.config = config
        self.global_step = 0
        self._train_iter: Iterator | None = None
        self._init_tracking()

    def _init_tracking(self) -> None:
        """Initialize experiment tracking (e.g., Weights & Biases)."""
        wandb_config = self.config.wandb_config

        if not wandb_config.enabled:
            logger.info("WandB logging is disabled in configuration.")
            return

        if wandb is None:
            logger.warning("WandB library not installed. Logging disabled. Run: pip install wandb")
            self.config.wandb_config.enabled = False
            return

        logger.info(f"Initializing WandB project: {wandb_config.project}")
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
                "advantage": asdict(self.config.advantage_config),
            },
        )
        logger.info(f"WandB run initialized: {wandb.run.name if wandb.run else 'Unknown'}")

    def _log_metrics(self, metrics: dict[str, float], step: int | None = None, prefix: str = "train") -> None:
        """
        Log metrics to the tracking system.

        Args:
            metrics: Dictionary of metric names and values.
            step: Current training step (defaults to self.global_step).
            prefix: Prefix to prepend to metric names (e.g., "train", "eval").
        """
        if not self.config.wandb_config.enabled or wandb is None:
            return

        step = step if step is not None else self.global_step
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(log_dict, step=step)

    def _get_next_batch(self) -> Sample:
        """
        Get the next batch from the training dataloader, cycling infinitely.

        Returns:
            A batch of training samples.
        """
        if self._train_iter is None:
            self._train_iter = iter(self.train_dataloader)

        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)

        return batch

    def _ensure_checkpoint_dir(self, name: str) -> Path:
        """
        Get the path for a checkpoint directory, creating it if necessary.

        Args:
            name: Name of the checkpoint subdirectory.

        Returns:
            Path object pointing to the checkpoint directory.
        """
        base_dir = Path.cwd() / "checkpoints"
        checkpoint_dir = base_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def save_eval_results(self, results: list[dict], step: int) -> None:
        """
        Save evaluation results to a JSONL file.

        Args:
            results: List of result dictionaries.
            step: Current training step.
        """
        import json

        results_dir = Path.cwd() / "eval_results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / f"step_{step}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(results)} evaluation results to {output_file}")

    @abstractmethod
    def train(self) -> None:
        """
        Execute the main training loop.

        This method should handle the entire training process containing:
        - Rollout generation
        - Advantage computation
        - Optimization steps
        - Evaluation and checkpointing
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """
        Run evaluation on the evaluation dataset.

        Returns:
            A dictionary of aggregated evaluation metrics.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path | str) -> None:
        """
        Save the model checkpoint.

        Args:
            path: Target directory path for saving the checkpoint.
        """
        pass
