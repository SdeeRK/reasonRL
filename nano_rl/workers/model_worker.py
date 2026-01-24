import math

import ray
import torch
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM

from ..core.config import ModelConfig, TrainConfig
from ..core.types import ModelCheckpoint, StepMetrics, TrainingBatch


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a scheduler with linear warmup and cosine decay to min_lr_ratio * initial_lr.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a ratio of the initial lr (decays to lr * min_lr_ratio).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay to min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Cosine decay from 1.0 to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)


@ray.remote(num_gpus=1)
class ModelWorker:
    """
    PyTorch-based policy training worker implementing ModelWorkerProtocol.

    Responsible for:
    - Training the policy model with gradient updates
    - Providing weights for synchronization with RolloutWorker
    - Saving checkpoints (with LoRA merge support)

    Runs as a Ray remote actor with GPU access.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device("cuda")

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config.model_name_or_path,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Apply LoRA if enabled
        if self.model_config.use_lora:
            self.model = self._apply_lora()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
        )

        self.lr_scheduler = self._create_lr_scheduler()

        if self.train_config.gradient_checkpoint:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

    def _apply_lora(self):
        """Apply LoRA adapter to the model."""
        target_modules = self.model_config.lora_target_modules
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()
        return model

    def _create_lr_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler based on configuration."""
        num_training_steps = self.train_config.n_grpo_steps
        num_warmup_steps = int(num_training_steps * self.train_config.warmup_ratio)

        if self.train_config.lr_scheduler_type == "constant":
            # Constant learning rate with optional warmup
            def lr_lambda(current_step: int) -> float:
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return 1.0

            return LambdaLR(self.optimizer, lr_lambda)
        else:
            # Warmup cosine decay scheduler
            return get_warmup_cosine_scheduler(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=self.train_config.min_lr_ratio,
            )

    @property
    def dtype(self):
        torch_type = self.model_config.dtype
        if torch_type == "bfloat16":
            return torch.bfloat16
        elif torch_type == "float16":
            return torch.float16
        elif torch_type == "float32":
            return torch.float32
        else:
            raise ValueError("noncorrect dtype")

    def train_step(self, batch: TrainingBatch) -> StepMetrics:
        pass

    def _restore_lora_and_optimizer(self) -> None:
        """Re-apply LoRA and restore optimizer after merge_and_unload."""
        self.model = self._apply_lora()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
        )
        self.lr_scheduler = self._create_lr_scheduler()

    def get_weights(self, global_step: int) -> ModelCheckpoint:
        """
        Get the full merged model weights for synchronization.
        If LoRA is enabled, merges LoRA weights into the base model and returns full weights.
        """
        if self.model_config.use_lora:
            merged_model = self.model.merge_and_unload()
            state_dict = {k: v.cpu() for k, v in merged_model.state_dict().items()}
            self._restore_lora_and_optimizer()
        else:
            state_dict = self.model.state_dict()

        return ModelCheckpoint(global_step=global_step, state_dict=state_dict)

    def save_checkpoint(self, path: str) -> None:
        """
        Save the model checkpoint.
        If LoRA is enabled, merges weights and saves the full model.
        """
        if self.model_config.use_lora:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(path)
            self._restore_lora_and_optimizer()
        else:
            self.model.save_pretrained(path)
