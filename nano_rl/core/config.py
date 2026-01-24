from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    gpu_memory_utilization: float = field(
        metadata={
            "help": "The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache."
        }
    )
    dtype: Literal["bfloat16", "float16", "float32"] = field(
        default="bfloat16",
        metadata={"help": "The data type for the model weights and activations."},
    )

    # Ray GPU allocation
    model_num_gpus: int = field(
        default=1,
        metadata={"help": "Number of GPUs to allocate for ModelWorker (training)."},
    )
    rollout_num_gpus: int = field(
        default=1,
        metadata={
            "help": "Number of GPUs to allocate for RolloutWorker (vLLM inference)."
        },
    )

    # LoRA configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA (Low-Rank Adaptation) for training."},
    )
    lora_rank: int = field(
        default=8,
        metadata={
            "help": "LoRA rank (r). Higher values increase capacity but also memory usage."
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor. Typically set to 2*rank."},
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers."}
    )
    lora_target_modules: list[str] | None = field(
        default=None,
        metadata={
            "help": "List of module names to apply LoRA to. If None, defaults to ['q_proj', 'v_proj']."
        },
    )


@dataclass
class GenerationConfig:
    group_size: int = field(
        default=1, metadata={"help": "Number of samples to generate per prompt."}
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
    min_tokens: int = field(
        default=4, metadata={"help": "Minimum number of tokens to generate."}
    )
    max_tokens: int = field(
        default=512, metadata={"help": "The maximum number of tokens to generate."}
    )
    include_logprobs: bool = field(
        default=True,
        metadata={
            "help": "Whether to return log probabilities of the generated tokens."
        },
    )


@dataclass
class TrainConfig:
    # GRPO specific parameters
    n_grpo_steps: int = field(
        default=200, metadata={"help": "Number of GRPO steps to run."}
    )

    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum total sequence length (prompt + response) for training. Sequences exceeding this will be truncated from the end."
        },
    )

    learning_rate: float = field(
        default=1e-5, metadata={"help": "Learning rate for the policy optimization."}
    )

    advantage_eps: float = field(
        default=1e-6,
        metadata={"help": "Small epsilon for advantage calculation stability."},
    )

    rollout_batch_size: int = field(
        default=256,
        metadata={
            "help": "Total number of samples generated per rollout step (group_size * n_prompts)."
        },
    )

    # Training Loop Parameters
    epochs_per_rollout_batch: int = field(
        default=1,
        metadata={
            "help": "Number of training epochs per collected rollout batch (1 for online/on-policy)."
        },
    )

    train_batch_size: int = field(
        default=256,
        metadata={
            "help": "Batch size for training updates (should equal rollout_batch_size for on-policy)."
        },
    )

    micro_train_batch_size: int = field(
        default=2, metadata={"help": "Size of micro-batch per device step."}
    )

    # Algorithm specifics
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = field(
        default="reinforce_with_baseline",
        metadata={"help": "Type of loss function to use."},
    )

    use_std_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages using standard deviation."},
    )

    # Optimizer (AdamW) specific
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW optimizer."}
    )

    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer."}
    )

    adam_beta2: float = field(
        default=0.95, metadata={"help": "Beta2 for AdamW optimizer."}
    )

    gradient_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory."},
    )

    grad_norm_clip: float = field(
        default=1.0, metadata={"help": "Gradient norm clipping value."}
    )

    # Learning rate scheduler parameters
    lr_scheduler_type: Literal["constant", "cosine"] = field(
        default="cosine",
        metadata={
            "help": "Type of learning rate scheduler: 'constant' or 'cosine' (warmup cosine decay)."
        },
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={
            "help": "Ratio of total training steps to use for linear warmup (0.0 to 1.0)."
        },
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

    @property
    def gradient_accumulation_steps(self) -> int:
        """Number of steps to accumulate gradients before backward pass."""
        return self.train_batch_size // self.micro_train_batch_size

    @property
    def n_microbatches_per_rollout_batch(self) -> int:
        """Number of microbatches needed to cover one rollout batch."""
        return self.rollout_batch_size // self.micro_train_batch_size


@dataclass
class RLConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        assert (
            self.train_config.rollout_batch_size % self.generation_config.group_size
            == 0
        ), "rollout_batch_size must be divisible by group_size"

        assert (
            self.train_config.train_batch_size >= self.generation_config.group_size
        ), "train_batch_size must be greater than or equal to group_size"
