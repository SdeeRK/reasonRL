import ray
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ..core.config import RLConfig
from ..core.interfaces import (
    AdvantageComputerProtocol,
    LossComputerProtocol,
    ModelWorkerProtocol,
    RewardComputerProtocol,
    RolloutWorkerProtocol,
)
from ..core.processor import DataProcessor
from ..core.types import PromptBatch, RewardBatch, RolloutBatch, Sample, StepMetrics
from ..logger import get_logger
from ..core.profiler import Profiler
from .base_trainer import BaseTrainer

try:
    import wandb
except ImportError:
    wandb = None

logger = get_logger(__name__)


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
        model_worker_cls: type[ModelWorkerProtocol],
        rollout_worker_cls: type[RolloutWorkerProtocol],
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name_or_path)
        self.data_processor = DataProcessor(
            tokenizer=self.tokenizer,
            truncate_max_length=config.train_config.truncate_max_length,
        )

        self.reward_fn = reward_fn
        self.advantage_fn = advantage_fn
        self.loss_fn = loss_fn
        self.profiler = Profiler()

        self.model_worker = model_worker_cls.options(num_gpus=config.model_config.model_num_gpus).remote(
            config.model_config,
            config.train_config,
            self.loss_fn,
        )
        self.rollout_worker = rollout_worker_cls.options(num_gpus=config.model_config.rollout_num_gpus).remote(
            config.model_config,
            config.generation_config,
        )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.n_prompts,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

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
        generation_config = self.config.generation_config
        wandb_config = self.config.wandb_config

        logger.info(f"Starting training for {train_config.n_grpo_steps} steps")

        for rl_step in range(train_config.n_grpo_steps):
            samples: list[Sample] = self._get_next_batch()
            logger.info(f"rl step {rl_step + 1}/{train_config.n_grpo_steps}")

            with self.profiler.timer("data_process"):
                prompt_batch: PromptBatch = self.data_processor.to_prompt_batch(samples)

            logger.info(f"rollout worker generate {len(samples)} prompts * {generation_config.group_size} group_size...")
            with self.profiler.timer("rollout_generate"):
                rollout_batch: RolloutBatch = ray.get(self.rollout_worker.generate.remote(prompt_batch, generation_config))

            logger.info("computing old log probs...")
            with self.profiler.timer("compute_ref_logprobs"):
                rollout_batch: RolloutBatch = ray.get(self.model_worker.compute_log_probs.remote(rollout_batch))

            with self.profiler.timer("reward_compute"):
                reward_batch: RewardBatch = self.reward_fn(rollout_batch)
                rollout_summary = rollout_batch.summary()
                rollout_summary.update(reward_batch.summary())
            
            logger.info(f"rollout summary : {rollout_summary}")
            self._log_metrics(rollout_summary, prefix="rollout")

            with self.profiler.timer("advantage_compute"):
                advantage_samples = self.advantage_fn(reward_batch)
                training_batch = self.data_processor.to_training_batch(advantage_samples)

            with self.profiler.timer("train_step"):
                for epoch in range(train_config.epochs_per_rollout_batch):
                    for mini_batch_step, mini_batch in enumerate(training_batch.minibatches(train_config.train_batch_size)):
                        self.global_step += 1

                        ray.get(self.model_worker.zero_grad.remote())
                        metrics: StepMetrics = ray.get(self.model_worker.train_step.remote(mini_batch))
                        grad_norm: float = ray.get(self.model_worker.optimizer_step.remote())
                        metrics.gradient_norm = grad_norm

                        logger.info(
                            f"global step {self.global_step} | rl step {rl_step + 1} | epoch {epoch + 1} | batch {mini_batch_step + 1} | "
                            f"loss: {metrics.loss:.3e} | "
                            f"lr: {metrics.learning_rate:.3e} | "
                            f"grad: {grad_norm:.4f} | "
                            f"ratio: {metrics.stats.get('mean_ratio', 0.0):.4f}"
                        )

                        train_metrics = {
                            "loss": metrics.loss,
                            "learning_rate": metrics.learning_rate,
                            "gradient_norm": grad_norm,
                            **metrics.stats,
                        }
                        if metrics.entropy is not None:
                            train_metrics["entropy"] = metrics.entropy
                        self._log_metrics(train_metrics, prefix="train")

            logger.info("sync policy and rollout model weights...")
            with self.profiler.timer("weight_sync"):
                checkpoint_ref = self.model_worker.get_weights.remote(self.global_step)
                ray.get(self.rollout_worker.update_weights.remote(checkpoint_ref))
            
            self.profiler.log_stats()

            if self.eval_dataset and (rl_step + 1) % wandb_config.eval_interval == 0:
                logger.info(f"starting evaluation at global step {self.global_step}...")
                eval_metrics = self.evaluate()  # BaseTrainer calls evaluate()
                self._log_metrics(eval_metrics, prefix="eval")
                logger.info(f"evaluation done. Metrics: {eval_metrics}")

            if (rl_step + 1) % wandb_config.save_interval == 0:
                path = self._ensure_checkpoint_dir(f"checkpoint-step-{self.global_step}")
                self.save_checkpoint(path)

        path = self._ensure_checkpoint_dir("final_model")
        self.save_checkpoint(path)

        if wandb_config.enabled and wandb is not None:
            wandb.finish()

        logger.info("training completed!")

    def evaluate(self) -> dict[str, float]:
        """
        Run evaluation on eval_dataset.

        Loads all prompts into memory and runs a single large batch generation.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}

        logger.info("running evaluation...")

        samples = [self.eval_dataset[i] for i in range(len(self.eval_dataset))]

        import copy

        eval_gen_config = copy.deepcopy(self.config.generation_config)
        eval_gen_config.group_size = 1

        prompt_batch: PromptBatch = self.data_processor.to_prompt_batch(samples)
        rollout_batch: RolloutBatch = ray.get(self.rollout_worker.generate.remote(prompt_batch, eval_gen_config))
        reward_batch: RewardBatch = self.reward_fn(rollout_batch)
        metrics = reward_batch.summary()

        lengths = []
        samples_to_save = []

        has_info = reward_batch.reward_details is not None

        for i, (group, group_rewards) in enumerate(zip(reward_batch.groups, reward_batch.rewards)):
            reward = group_rewards[0]
            rollout = group.items[0]
            lengths.append(len(rollout.response_ids))

            item_metrics = {}
            if has_info:
                item_metrics = reward_batch.reward_details[i][0]

            samples_to_save.append(
                {
                    "prompt": group.sample.prompt,
                    "response": rollout.response_text,
                    "ground_truth": group.sample.ground_truth,
                    "reward": reward,
                    "response_length": len(rollout.response_ids),
                    **item_metrics,
                }
            )

        self.save_eval_results(samples_to_save, self.global_step)

        metrics.update(rollout_batch.summary())

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk."""
        ray.get(self.model_worker.save_checkpoint.remote(str(path)))
        logger.info(f"checkpoint saved to {path}")
