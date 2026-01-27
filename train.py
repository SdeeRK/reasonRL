import sys
from dataclasses import dataclass
from pathlib import Path

import ray
import tyro

from reason_rl.advantage import GRPOAdvantage
from reason_rl.core.config import RLConfig
from reason_rl.data import PromptDataset
from reason_rl.logger import get_logger, setup_logger
from reason_rl.loss import GRPOLoss
from reason_rl.reward import MathReward
from reason_rl.trainer import GRPOTrainer
from reason_rl.workers import ModelWorker, RolloutWorker

logger = get_logger(__name__)


def init_ray(working_dir: str = "reason_rl"):
    """Initialize Ray with appropriate runtime environment settings."""
    if not ray.is_initialized():
        logger.info(f"Initializing Ray with working_dir='{working_dir}'...")
        ray.init(
            runtime_env={
                "working_dir": working_dir,
                "excludes": [
                    "**/__pycache__/**",
                    "**/*.pyc",
                    "**/checkpoint_/**",
                    "**/.git/**",
                ],
            }
        )
    else:
        logger.info("Ray is already initialized.")


@dataclass
class Args:
    """Command line arguments for GRPO training."""

    config: Path
    """Path to the configuration YAML file."""


def main(args: Args):
    """Main training entry point."""
    setup_logger()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    init_ray(working_dir="reason_rl")

    logger.info(f"Loading config from {args.config}...")
    config = RLConfig.from_yaml(str(args.config.resolve()))
    logger.info("config:\n", config)

    model_path = Path(config.model_config.model_name_or_path)
    if model_path.exists():
        absolute_model_path = model_path.resolve()
        logger.info(f"Resolved absolute model path: {absolute_model_path}")
        config.model_config.model_name_or_path = str(absolute_model_path)

    logger.info("Initializing datasets...")
    train_dataset = PromptDataset(config.train_config.train_dataset)

    eval_dataset = None
    if config.train_config.eval_dataset:
        logger.info(f"Initializing eval dataset from {config.train_config.eval_dataset}...")
        eval_dataset = PromptDataset(config.train_config.eval_dataset)

    logger.info("Initializing reward, advantage, and loss functions...")
    reward_fn = MathReward()
    advantage_fn = GRPOAdvantage(config=config.advantage_config)
    loss_fn = GRPOLoss(config=config.loss_config)

    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        config=config,
        reward_fn=reward_fn,
        advantage_fn=advantage_fn,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_worker_cls=ModelWorker,
        rollout_worker_cls=RolloutWorker,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.success("Training completed successfully!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
