import os
from collections.abc import Iterator

import ray
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from ..core.config import RLConfig
from ..core.interfaces import (
    AdvantageComputerProtocol,
    LossComputerProtocol,
    RewardComputerProtocol,
)
from ..core.processor import DataProcessor
from ..core.types import Sample, TrainingBatch
from ..workers.model_worker import ModelWorker
from ..workers.rollout_worker import RolloutWorker
from .base_trainer import BaseTrainer


try:
    import wandb
except ImportError:
    wandb = None


class GRPOTrainer(BaseTrainer):
    """
    GRPO training orchestrator.

    Initializes and coordinates all components:
    - RolloutWorker: Generate responses via vLLM
    - ModelWorker: Train policy via PyTorch
    - reward_fn / advantage_fn: Compute rewards and advantages
    - DataProcessor: Handle tokenization and padding
    - WandB: Experiment tracking and logging
    """

    def __init__(
        self,
        config: RLConfig,
        reward_fn: RewardComputerProtocol,
        advantage_fn: AdvantageComputerProtocol,
        loss_fn: LossComputerProtocol,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_config.model_name_or_path
        )
        self.data_processor = DataProcessor(
            tokenizer=self.tokenizer,
            truncate_max_length=config.train_config.truncate_max_length,
        )

        self.reward_fn = reward_fn
        self.advantage_fn = advantage_fn
        self.loss_fn = loss_fn

        self.model_worker = ModelWorker.options(
            num_gpus=config.model_config.model_num_gpus
        ).remote(
            config.model_config,
            config.train_config,
            self.loss_fn,
        )
        self.rollout_worker = RolloutWorker.options(
            num_gpus=config.model_config.rollout_num_gpus
        ).remote(
            config.model_config,
            config.generation_config,
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._compute_batch_size(),
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

        self.global_step = 0
        self._train_iter: Iterator | None = None

        self._init_wandb()

    def _compute_batch_size(self) -> int:
        """Compute number of prompts per batch based on rollout_batch_size and group_size."""
        batch_size = (
            self.config.train_config.rollout_batch_size
            // self.config.generation_config.group_size
        )
        logger.info(
            f"Computed batch size: {batch_size} "
            f"(rollout_batch_size={self.config.train_config.rollout_batch_size}, "
            f"group_size={self.config.generation_config.group_size})"
        )
        return batch_size

    def train(self) -> None:
        """
        Main training loop.

        Runs for n_grpo_steps iterations, each consisting of:
        1. Sample batch from dataloader
        2. Generate rollouts via RolloutWorker
        3. Compute rewards and advantages
        4. Train policy for epochs_per_rollout_batch epochs
        5. Sync weights to RolloutWorker
        """
        train_config = self.config.train_config
        wandb_config = self.config.wandb_config

        logger.info(f"Starting training for {train_config.n_grpo_steps} steps")

        for step in tqdm(
            range(train_config.n_grpo_steps), desc="Training Steps", dynamic_ncols=True
        ):
            self.global_step = step

            # 1. Get batch of samples
            samples = self._get_next_batch()
            logger.info(
                f"Step {step + 1}/{train_config.n_grpo_steps}: Got batch of {len(samples)} samples"
            )

            # 2. Tokenize for vLLM (no padding)
            prompt_batch = self.data_processor.to_prompt_batch(samples)

            # 3. Generate rollouts
            rollout_batch = ray.get(
                self.rollout_worker.generate.remote(
                    prompt_batch, self.config.generation_config
                )
            )

            # 4. Compute rewards
            reward_batch = self.reward_fn(rollout_batch)

            # Log rollout statistics
            rollout_stats = self._compute_rollout_stats(reward_batch)
            logger.info(f"Step {step + 1}: Rollout stats: {rollout_stats}")

            # 5. Compute advantages
            advantage_samples = self.advantage_fn(reward_batch)

            # 6. Convert to training tensors (padding)
            training_batch = self.data_processor.to_training_batch(advantage_samples)

            # 7. Inner training loop (epochs per rollout batch)
            # logger.info(f"Step {step+1}: Starting inner training loop ({train_config.epochs_per_rollout_batch} epochs)")
            batch_size = train_config.train_batch_size

            for epoch in tqdm(
                range(train_config.epochs_per_rollout_batch),
                desc="Inner Epochs",
                leave=False,
                dynamic_ncols=True,
            ):
                # Iterate over minibatches
                for i, mini_batch in enumerate(training_batch.minibatches(batch_size)):
                    # Clear gradients
                    ray.get(self.model_worker.zero_grad.remote())

                    # Forward + backward (with gradient accumulation)
                    metrics = ray.get(self.model_worker.train_step.remote(mini_batch))

                    # Optimizer step + lr scheduler step
                    grad_norm = ray.get(self.model_worker.optimizer_step.remote())
                    metrics.gradient_norm = grad_norm

                    # Log metrics
                    self._log_metrics(metrics, step, epoch, rollout_stats)
                    logger.info(
                        f"Step {step + 1} Epoch {epoch + 1} Batch {i + 1}: "
                        f"Loss={metrics.loss:.4f} "
                        f"LR={metrics.learning_rate:.6f} "
                        f"GradNorm={grad_norm:.4f}"
                    )

            # 8. Sync weights to rollout worker
            checkpoint = ray.get(self.model_worker.get_weights.remote(self.global_step))
            ray.get(self.rollout_worker.update_weights.remote(checkpoint))

            # 9. Periodic evaluation
            if self.eval_dataset and (step + 1) % wandb_config.eval_interval == 0:
                logger.info(f"Step {step + 1}: Starting evaluation...")
                eval_metrics = self.evaluate(step=step + 1)
                self._log_eval_metrics(eval_metrics, step)
                logger.info(
                    f"Step {step + 1}: Evaluation done. Metrics: {eval_metrics}"
                )

            # 10. Periodic checkpoint saving
            if (step + 1) % wandb_config.save_interval == 0:
                path = self._get_checkpoint_path(f"checkpoint-step-{step + 1}")
                self.save_checkpoint(path)

        # Save final checkpoint
        path = self._get_checkpoint_path("final_model")
        self.save_checkpoint(path)

        # Finish WandB run
        if wandb_config.enabled and wandb is not None:
            wandb.finish()

        logger.info("Training completed!")

    def _compute_rollout_stats(self, reward_batch) -> dict[str, float]:
        """Compute statistics from rollout batch."""
        all_rewards = []
        total_response_length = 0
        num_responses = 0

        for group, group_rewards in zip(reward_batch.groups, reward_batch.rewards):
            all_rewards.extend(group_rewards)
            for rollout in group.items:
                total_response_length += len(rollout.response_ids)
                num_responses += 1

        import statistics

        return {
            "rollout/mean_reward": statistics.mean(all_rewards) if all_rewards else 0.0,
            "rollout/std_reward": statistics.stdev(all_rewards)
            if len(all_rewards) > 1
            else 0.0,
            "rollout/max_reward": max(all_rewards) if all_rewards else 0.0,
            "rollout/min_reward": min(all_rewards) if all_rewards else 0.0,
            "rollout/mean_response_length": total_response_length
            / max(1, num_responses),
            "rollout/num_samples": num_responses,
        }

    def evaluate(self, step: int = 0) -> dict[str, float]:
        """
        Run evaluation on eval_dataset.

        Loads all prompts into memory and runs a single large batch generation.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}

        logger.info("Running evaluation...")

        # 1. Load all samples directly from the injected dataset
        samples = [self.eval_dataset[i] for i in range(len(self.eval_dataset))]

        # 2. Create temp config with group_size=1
        # We copy the config to avoid modifying the training config
        import copy

        eval_gen_config = copy.deepcopy(self.config.generation_config)
        eval_gen_config.group_size = 1

        # 3. Generate all responses in one go (let vLLM handle batching)
        prompt_batch = self.data_processor.to_prompt_batch(samples)

        # Ray remote call for generation
        rollout_batch = ray.get(
            self.rollout_worker.generate.remote(prompt_batch, eval_gen_config)
        )

        # 4. Compute rewards
        reward_batch = self.reward_fn(rollout_batch)

        # 5. Compute metrics
        all_rewards = []
        total_response_length = 0
        num_correct = 0
        num_total = len(samples)

        # Prepare for saving results
        samples_to_save = []

        for group, group_rewards in zip(reward_batch.groups, reward_batch.rewards):
            # Since group_size=1, we only have one item per group
            reward = group_rewards[0]
            rollout = group.items[0]

            all_rewards.append(reward)
            total_response_length += len(rollout.response_ids)

            # Check if correct (reward > threshold, assuming 1.0 is correct/0.5 threshold)
            if reward > 0.5:
                num_correct += 1

            # Collect individual result
            samples_to_save.append(
                {
                    "prompt": group.sample.prompt,
                    "response": rollout.response_text,
                    "ground_truth": group.sample.ground_truth,
                    "reward": reward,
                }
            )

        # Save results using helper from BaseTrainer
        self.save_eval_results(samples_to_save, step)

        import statistics

        metrics = {
            "eval/mean_reward": statistics.mean(all_rewards) if all_rewards else 0.0,
            "eval/std_reward": statistics.stdev(all_rewards)
            if len(all_rewards) > 1
            else 0.0,
            "eval/max_reward": max(all_rewards) if all_rewards else 0.0,
            "eval/min_reward": min(all_rewards) if all_rewards else 0.0,
            "eval/accuracy": num_correct / max(1, num_total),
            "eval/mean_response_length": total_response_length / max(1, num_total),
            "eval/num_samples": num_total,
        }

        logger.info(
            f"Eval results: mean_reward={metrics['eval/mean_reward']:.4f}, "
            f"accuracy={metrics['eval/accuracy']:.4f}"
        )

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        ray.get(self.model_worker.save_checkpoint.remote(path))
        logger.info(f"Checkpoint saved to {path}")

        # Log checkpoint to WandB as artifact (optional)
        if self.config.wandb_config.enabled and wandb is not None:
            artifact = wandb.Artifact(
                name=f"model-{self.global_step}",
                type="model",
                description=f"Model checkpoint at step {self.global_step}",
            )
            artifact.add_dir(path)
            wandb.log_artifact(artifact)
