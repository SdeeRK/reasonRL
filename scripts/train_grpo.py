import os
import sys
import argparse
import logging
import ray
from loguru import logger

from nano_rl.core.config import RLConfig
from nano_rl.data.dataset import PromptDataset
from nano_rl.reward.math_reward import MathReward
from nano_rl.advantage.grpo_advantage import GRPOAdvantage
from nano_rl.loss.grpo_loss import GRPOLoss
from nano_rl.trainer.grpo_trainer import GRPOTrainer

# # Configure logger with a concise format
# logger.remove()
# logger.add(
#     sys.stderr,
#     format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
#     level="INFO"
# )


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config yaml file"
    )
    args = parser.parse_args()

    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # 2. Load Config
    logger.info(f"Loading config from {args.config}...")
    config = RLConfig.from_yaml(args.config)

    # 3. Initialize Datasets
    logger.info("Initializing datasets...")
    train_dataset = PromptDataset(config.train_config.train_dataset)

    eval_dataset = None
    if config.train_config.eval_dataset:
        eval_dataset = PromptDataset(config.train_config.eval_dataset)

    # 4. Initialize Core Components
    logger.info("Initializing reward, advantage, and loss functions...")
    reward_fn = MathReward()
    advantage_fn = GRPOAdvantage(config=config.advantage_config)
    loss_fn = GRPOLoss(config=config.loss_config)

    # 5. Initialize Trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        config=config,
        reward_fn=reward_fn,
        advantage_fn=advantage_fn,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 6. Start Training
    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
