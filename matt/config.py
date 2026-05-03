from dataclasses import dataclass


@dataclass
class MATTConfig:
    """Configuration for MATT (Model-Aware Tokenizer Transfer) training.

    This dataclass contains all parameters for training a tokenizer transfer model
    using Attention Influence Modeling (AIM). It controls training dynamics,
    hardware settings, and logging behavior.

    Attributes:
        seed: Random seed for reproducibility.
        accelerator: Hardware accelerator type ("gpu", "cpu", "tpu", etc.).
        devices: Number of devices to use (-1 for all available).
        checkpoint_output_dir: Directory to save model checkpoints.
        dataset_output_dir: Directory for cached preprocessed datasets.

        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.

        max_epochs: Maximum number of training epochs.
        max_steps: Maximum number of training steps (-1 for no limit).
        gradient_clip_val: Maximum gradient norm for clipping.
        gradient_clip_algorithm: Algorithm for gradient clipping ("norm" or "value").
        accumulate_grad_batches: Batches to accumulate before optimizer step.
        pad_to_multiple_of: Pad sequence lengths to multiple of this value.
        strategy: PyTorch Lightning training strategy ("auto", "ddp", etc.).
        precision: Training precision ("bf16-mixed", "16-mixed", "32", etc.).
        log_every_n_steps: Frequency of logging metrics.
        save_every_n_steps: Frequency of saving checkpoints.
        save_top_k: Number of best checkpoints to keep (-1 for all).
        loss: Loss function ("mse" or "cosine").
        aim_impl: AIM implementation variant ("aim" or "aim_star").
        lr: Learning rate for optimizer.
        num_warmup_steps: Linear warmup steps for WSD scheduler (0 = no warmup).
        num_decay_steps: Linear decay steps at end of training (0 = no decay).
        with_ntp: Enable Next Token Prediction loss alongside AIM loss. Required
            for models with untied input/output embeddings (e.g. Qwen3). The
            student model must be created with with_ntp=True as well.

        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_output_dir: Directory for W&B logs.
    """
    # Core settings
    seed: int = 0
    accelerator: str = "gpu"
    devices: int | str = -1
    checkpoint_output_dir: str = "output/checkpoints"
    dataset_output_dir: str = "output/datasets"

    # Data settings
    batch_size: int = 4
    num_workers: int = 4

    # Training settings
    max_epochs: int = 1
    max_steps: int = -1
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    accumulate_grad_batches: int = 1
    pad_to_multiple_of: int = 8
    strategy: str = "auto"
    precision: str = "bf16-mixed"
    log_every_n_steps: int = 50
    save_every_n_steps: int = 50_000
    save_top_k: int = -1
    loss: str = "mse"
    aim_impl: str = "aim"
    lr: float = 1e-4
    num_warmup_steps: int = 0
    num_decay_steps: int = 0
    with_ntp: bool = False

    # Logging settings
    use_wandb: bool = False
    wandb_project: str = "matt"
    wandb_output_dir: str = "output/wandb"
