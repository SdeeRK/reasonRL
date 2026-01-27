import math

import ray
import torch
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.config import ModelConfig, TrainConfig
from ..core.interfaces import LossComputerProtocol
from ..core.types import ModelCheckpoint, RolloutBatch, StepMetrics, TrainingBatch
from ..logger import get_logger

logger = get_logger(__name__)


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
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # Cosine decay from 1.0 to min_lr_ratio
        progress = min(1.0, progress)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

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

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        loss_fn: LossComputerProtocol,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.loss_fn = loss_fn
        self.device = torch.device("cuda")

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=self.dtype,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
        )

        if self.model_config.use_lora:
            self.model = self._apply_lora()
            self.model.enable_input_require_grads()
            logger.info("model worker loratrainable parameters below:")
            self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
        )

        self.lr_scheduler = self._create_lr_scheduler()

        if self.train_config.gradient_checkpoint:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logger.info("model worker enable gradient checkpointing")

    @torch.inference_mode()
    def compute_log_probs(self, rollout_batch: RolloutBatch) -> RolloutBatch:
        self.model.eval()
        for group in rollout_batch.groups:
            prompt_ids = group.prompt_ids
            for single_rollout in group.items:
                response_ids = single_rollout.response_ids
                full_ids = prompt_ids + response_ids

                # shape: [1, seq_len]
                input_tensor = torch.tensor([full_ids], device=self.device)
                outputs = self.model(input_tensor)

                # shape: [1, seq_len, vocab_size]
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_tensor[..., 1:].contiguous()

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

                # token_log_probs shape: [1, seq_len-1]
                token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

                # 只截取 Response 部分
                # Prompt 长度为 len(prompt_ids)
                # 因为 shift 之后，下标 0 对应的是原来的 index 1 (Prompt 的第 2 个词)
                # 所以 Response 的第 1 个词对应的预测，位于 index = len(prompt_ids) - 1
                start_idx = len(prompt_ids) - 1
                response_log_probs = token_log_probs[0, start_idx:]

                single_rollout.old_log_probs = response_log_probs.tolist()

        return rollout_batch

    @property
    def dtype(self) -> torch.dtype:
        torch_type = self.model_config.dtype
        if torch_type == "bfloat16":
            return torch.bfloat16
        elif torch_type == "float16":
            return torch.float16
        elif torch_type == "float32":
            return torch.float32
        else:
            raise ValueError("noncorrect dtype")

    def zero_grad(self) -> None:
        """Clear gradients. Called by Trainer before train_step."""
        self.optimizer.zero_grad()

    def optimizer_step(self) -> float:
        """
        Perform optimizer step and lr scheduler step.
        Called by Trainer after train_step.
        Returns:
            Gradient norm (if calculated during clipping, else 0.0).
        """
        grad_norm = 0.0
        # Gradient clipping (if configured)
        if self.train_config.max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm,
            ).item()

        self.optimizer.step()
        self.lr_scheduler.step()

        return grad_norm

    def _get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.lr_scheduler.get_last_lr()[0]

    def train_step(self, batch: TrainingBatch) -> StepMetrics:
        """
        Perform forward pass and backward pass with gradient accumulation.
        Does NOT call optimizer.step() - that's controlled by Trainer.

        Args:
            batch: Training batch containing input_ids, response_mask, old_log_probs, advantages.

        Returns:
            StepMetrics with loss and optional entropy and learning rate.
        """
        self.model.train()

        total_loss = 0.0
        total_entropy = 0.0 if self.train_config.return_token_entropy else None
        learning_rate = self._get_learning_rate()
        accumulated_stats: dict[str, float] = {}

        pbar = tqdm(range(self.train_config.gradient_accumulation_steps), desc="Gradient Acc", leave=False)
        for gradient_step in pbar:
            start_index = gradient_step * self.train_config.micro_train_batch_size
            end_index = start_index + self.train_config.micro_train_batch_size
            micro_batch = batch[start_index:end_index].to(self.device)

            # Shift for causal LM: input is [:-1], label is [1:]
            input_ids = micro_batch.input_ids[:, :-1]
            attention_mask = micro_batch.attention_mask[:, :-1]
            labels = micro_batch.input_ids[:, 1:]
            response_mask = micro_batch.response_mask[:, 1:]

            # Also shift old_log_probs to match labels
            old_log_probs = micro_batch.old_log_probs[:, 1:] if micro_batch.old_log_probs is not None else None
            advantages = micro_batch.advantages  # (batch_size,) - per sequence, no shift needed

            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss_metrics = self.loss_fn(
                model_logits=model_output.logits,
                old_log_probs=old_log_probs,
                labels=labels,
                response_mask=response_mask,
                advantages=advantages,
                return_token_entropy=self.train_config.return_token_entropy,
            )

            loss = loss_metrics.loss / self.train_config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()
            if self.train_config.return_token_entropy and loss_metrics.entropy is not None:
                total_entropy += loss_metrics.entropy

            loss_val = loss.item()
            postfix = {"loss": f"{loss_val:.3e}", "entropy": loss_metrics.entropy}
            pbar.set_postfix(postfix)

            # Accumulate stats
            for key, value in loss_metrics.stats.items():
                if key not in accumulated_stats:
                    accumulated_stats[key] = 0.0
                accumulated_stats[key] += value

        # Average stats across gradient accumulation steps
        num_steps = self.train_config.gradient_accumulation_steps
        averaged_stats = {k: v / num_steps for k, v in accumulated_stats.items()}

        # Compute average entropy if enabled
        entropy = None
        if self.train_config.return_token_entropy and total_entropy is not None:
            entropy = total_entropy / num_steps

        return StepMetrics(
            loss=total_loss,
            entropy=entropy,
            stats=averaged_stats,
            learning_rate=learning_rate,
        )

    def get_weights(self, global_step: int) -> ModelCheckpoint:
        """
        Get the full merged model weights for synchronization.
        If LoRA is enabled, merges LoRA weights into the base model and returns full weights.
        """
        if self.model_config.use_lora:
            self.model.merge_adapter()
            underlying_model = self.model.base_model.model

            state_dict = {}
            for k, v in underlying_model.state_dict().items():
                if "lora_" in k or "adapter_" in k:
                    continue

                new_key = k.replace(".base_layer", "")
                state_dict[new_key] = v.cpu()

            self.model.unmerge_adapter()
        else:
            state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        return ModelCheckpoint(global_step=global_step, state_dict=state_dict)

    def save_checkpoint(self, path: str) -> None:
        """
        Save the model and tokenizer.
        Note: If LoRA is enabled, only the adapter weights are saved.

        To restore the full model, use the following logic:
        >>> from peft import PeftModel
        >>> from transformers import AutoModelForCausalLM
        >>> base = AutoModelForCausalLM.from_pretrained(base_model_path)
        >>> model = PeftModel.from_pretrained(base, lora_path)
        >>> model = model.merge_and_unload()
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

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
        num_training_steps = int(
            self.train_config.n_grpo_steps
            * self.train_config.epochs_per_rollout_batch
            * self.train_config.rollout_batch_size
            / self.train_config.train_batch_size
        )
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
