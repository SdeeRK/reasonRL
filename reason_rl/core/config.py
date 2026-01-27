from dataclasses import dataclass, field
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Math-1.5B",
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={
            "help": "The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache."
        },
    )
    dtype: Literal["bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "The data type for the model weights and activations."},
    )

    # Ray GPU allocation
    model_num_gpus: float = field(
        default=1,
        metadata={"help": "Number of GPUs to allocate for ModelWorker (training)."},
    )
    rollout_num_gpus: float = field(
        default=1,
        metadata={"help": "Number of GPUs to allocate for RolloutWorker (vLLM inference)."},
    )

    # LoRA configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA (Low-Rank Adaptation) for training."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank (r). Higher values increase capacity but also memory usage."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor. Typically set to 2*rank."},
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout probability for LoRA layers."})
    lora_target_modules: list[str] | None = field(
        default=None,
        metadata={"help": "List of module names to apply LoRA to. If None, defaults to ['q_proj', 'v_proj']."},
    )


@dataclass
class GenerationConfig:
    group_size: int = field(
        default=1,
        metadata={"help": "Number of samples to generate per prompt. when eval this will be 1"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.9,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."
        },
    )
    min_tokens: int = field(default=4, metadata={"help": "Minimum number of tokens to generate."})
    max_tokens: int = field(default=512, metadata={"help": "The maximum number of tokens to generate."})
    stop_tokens: list[str] | None = field(
        default=None,
        metadata={"help": "List of strings that stop the generation when they are generated."},
    )


@dataclass
class AdvantageConfig:
    mode: Literal["raw", "grpo", "grpo_no_std"] = field(
        default="grpo",
        metadata={
            "help": (
                "Advantage computation mode. "
                "'raw': use reward directly as advantage (no normalization). "
                "'grpo': subtract mean and divide by std within each group. "
                "'grpo_no_std': only subtract mean within each group."
            )
        },
    )
    advantage_eps: float = field(
        default=1e-6,
        metadata={"help": "Small epsilon for numerical stability when dividing by std."},
    )


@dataclass
class LossConfig:
    loss_level: Literal["token", "sequence"] = field(
        default="token",
        metadata={
            "help": (
                "Granularity of loss aggregation. "
                "'token': average loss over all response tokens across the batch. "
                "'sequence': first average loss per sequence, then average across sequences in the batch."
            )
        },
    )

    clip_range_left: float = field(
        default=0.2,
        metadata={"help": "Lower bound for the clipping range."},
    )
    clip_range_right: float = field(
        default=0.3,
        metadata={"help": "Upper bound for the clipping range."},
    )


@dataclass
class TrainConfig:
    # GRPO specific parameters
    n_grpo_steps: int = field(default=200, metadata={"help": "Number of GRPO steps to run."})

    train_dataset: str = field(
        default="",
        metadata={"help": "Path to the training dataset."},
    )
    eval_dataset: str = field(
        default="",
        metadata={"help": "Path to the evaluation dataset."},
    )

    truncate_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum total sequence length (prompt + response) for training. Sequences exceeding this will be truncated from the end."
        },
    )

    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate for the policy optimization."})

    rollout_batch_size: int = field(
        default=256,
        metadata={"help": "Total number of samples generated per rollout step (group_size * n_prompts)."},
    )

    # Training Loop Parameters
    epochs_per_rollout_batch: int = field(
        default=1,
        metadata={"help": "Number of training epochs per collected rollout batch (1 for online/on-policy)."},
    )

    train_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for training updates (should equal rollout_batch_size for on-policy)."},
    )

    micro_train_batch_size: int = field(default=2, metadata={"help": "Size of micro-batch per device step."})

    # Optimizer (AdamW) specific
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW optimizer."})

    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer."})

    adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW optimizer."})

    gradient_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory."},
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping."})

    return_token_entropy: bool = field(
        default=False,
        metadata={"help": "Whether to compute and return token-level entropy."},
    )

    # Learning rate scheduler parameters
    lr_scheduler_type: Literal["constant", "cosine"] = field(
        default="cosine",
        metadata={"help": "Type of learning rate scheduler: 'constant' or 'cosine' (warmup cosine decay)."},
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of total training steps to use for linear warmup (0.0 to 1.0)."},
    )

    min_lr_ratio: float = field(
        default=0.1,
        metadata={
            "help": "Minimum learning rate ratio relative to initial learning rate (scheduler decays to lr * min_lr_ratio)."
        },
    )

    def __post_init__(self):
        # Sanity checks
        assert self.train_batch_size >= self.micro_train_batch_size, (
            "train_batch_size must be greater than or equal to micro_train_batch_size"
        )
        assert self.train_batch_size % self.micro_train_batch_size == 0, (
            "train_batch_size must be divisible by micro_train_batch_size"
        )
        assert self.rollout_batch_size % self.train_batch_size == 0, (
            "rollout_batch_size must be divisible by train_batch_size"
        )

    @property
    def gradient_accumulation_steps(self) -> int:
        """Number of steps to accumulate gradients before backward pass."""
        return self.train_batch_size // self.micro_train_batch_size

    @property
    def n_microbatches_per_rollout_batch(self) -> int:
        """Number of microbatches needed to cover one rollout batch."""
        return self.rollout_batch_size // self.micro_train_batch_size


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = field(default=True, metadata={"help": "Whether to enable WandB logging."})

    project: str = field(default="reason-rl", metadata={"help": "WandB project name."})
    entity: str | None = field(
        default=None,
        metadata={"help": "WandB entity (username or team). If None, uses default."},
    )
    run_name: str | None = field(
        default=None,
        metadata={"help": "WandB run name. If None, auto-generated."},
    )
    tags: list[str] = field(
        default_factory=list,
        metadata={"help": "List of tags for the run."},
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Log metrics every N steps."},
    )
    eval_interval: int = field(
        default=100,
        metadata={"help": "Run evaluation every N steps."},
    )
    save_interval: int = field(
        default=100,
        metadata={"help": "Save checkpoint every N steps."},
    )


@dataclass
class RLConfig:
    model_config: ModelConfig
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    advantage_config: AdvantageConfig = field(default_factory=AdvantageConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        assert self.train_config.rollout_batch_size % self.generation_config.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )

        assert self.train_config.train_batch_size >= self.generation_config.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )

    @property
    def n_prompts(self) -> int:
        return self.train_config.rollout_batch_size // self.generation_config.group_size

    @classmethod
    def from_yaml(cls, path: str) -> "RLConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "RLConfig":
        from dacite import Config as DaciteConfig
        from dacite import from_dict as dacite_from_dict

        return dacite_from_dict(data_class=cls, data=data, config=DaciteConfig(strict=True))
