# MATT: Model-Aware Tokenizer Transfer

[![PyPI version](https://badge.fury.io/py/matt-tokenizer-transfer.svg)](https://badge.fury.io/py/matt-tokenizer-transfer)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

A Python library for transferring tokenizers between transformer models using **Attention Influence Modeling (AIM)**. MATT enables you to adapt a model trained with one tokenizer to use a completely different tokenizer while preserving its capabilities.

## What is MATT?

MATT (Model-Aware Tokenizer Transfer) adapts a pretrained model to use a different tokenizer while preserving its capabilities. It learns new embeddings by aligning segment-level attention patterns between the original (teacher) and new (student) tokenizations.

### Why MATT?

Existing tokenizer transfer methods fall into two camps, each with drawbacks:

- **Heuristic methods** (WECHSEL, FOCUS, TokAlign, Transtokenizers) initialize new embeddings from surface-level token similarity, ignoring how the model actually processes tokens. They recover roughly 70% of discriminative performance but only ~9% of generative performance.
- **Optimization-based methods** (NTP with frozen non-embedding weights) are effective but expensive: they require full forward and backward passes through the entire model at every training step.

MATT occupies an efficient middle ground. Its **Attention Influence Modeling (AIM)** objective distills inter-token communication patterns from the teacher model's attention layers directly into the student embeddings — a richer signal than heuristics at a fraction of the cost of full language modeling.

### How It Works

MATT uses **Attention Influence Modeling** to learn embeddings:

1. Tokenizes text with both teacher and student tokenizers
2. Finds minimal character-span segments consistent across both tokenizations using an offset-based algorithm (segments can split mid-word when both tokenizers agree)
3. Initializes new embeddings using FOCUS before training begins
4. Extracts attention weights and value states from the teacher model (only the first ~1/3 of layers are needed)
5. Trains student embeddings so their segment-level attention outputs match those of the teacher
6. Optionally freezes overlapping tokens for efficiency

## Paper

**[Model-Aware Tokenizer Transfer](https://arxiv.org/abs/2510.21954)**
Mykola Haltiuk, Aleksander Smywiński-Pohl
_arXiv preprint arXiv:2510.21954, 2025_

## Results (Gemma 3 12B PT → Ukrainian tokenizer)

Extended tokenizer: 262K → 387K vocab, compression rate 2.98 → 4.44, 1.5x inference and further training speedup (up to 1.8x on long sequences).
Evaluated on Belebele, Global MMLU (discriminative) and Long FLORES, WMT, XL-Sum (generative).

| Method | Training Time | Avg Disc | Avg Gen |
|--------|--------------|----------|---------|
| Gemma 3 12B PT (original) | — | 78.18 | 8.13 |
| FOCUS (heuristic) | — | 42.96 | 0.70 |
| Transtokenizers (heuristic) | — | 53.96 | 0.05 |
| FOCUS w/ NTP | 7h | 73.78 | 3.96 |
| **MATT** | **7h** | **77.27** | **6.45** |

MATT recovers ~99% of discriminative and ~79% of generative performance. The best NTP baseline at the same compute budget recovers ~94% discriminative but only ~49% generative.

See [paper](https://arxiv.org/abs/2510.11445) for more details and experiments.

## Features

- ✅ **5 Out-of-the-box Architectures**: Llama 3, Gemma3, Mistral, Qwen3, GPT-NeoX
- ✅ **Efficient Training**: Partial embedding freezing for overlapping tokens
- ✅ **Two AIM Variants**: Full (`aim`) and simplified (`aim_star`) implementations
- ✅ **NTP Loss**: Next Token Prediction loss for models with untied embeddings (e.g. Qwen3)
- ✅ **LR Scheduling**: Warmup-stable-decay scheduler for improved training dynamics
- ✅ **Production Ready**: PyTorch Lightning integration with distributed training
- ✅ **Monitoring**: Built-in VRAM, FLOPs, and timing callbacks
- ✅ **Extensible**: Easy to add new model architectures

## Installation

```bash
pip install matt-tokenizer-transfer
```

### Optional Dependencies

```bash
# For Weights & Biases logging
pip install matt-tokenizer-transfer[wandb]

# All optional dependencies
pip install matt-tokenizer-transfer[all]
```

## Quick Start

### Basic Usage

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from matt import MATT, MATTConfig
from matt.modeling import Gemma3ForAIM

# 1. Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
student_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 2. Configure training
config = MATTConfig(
    max_steps=10000,
    batch_size=4,
    lr=1e-4,
    precision="bf16-mixed",
)

# 3. Create MATT instance
matt = MATT(
    config=config,
    teacher_tokenizer=teacher_tokenizer,
    student_tokenizer=student_tokenizer,
)

# 4. Prepare dataset (cached to disk)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
matt.prepare_dataset(
    dataset=dataset,
    max_length=512,
    num_proc=8,
)

# 5. (Optional) Create frozen embeddings mask for efficiency
frozen_mask = matt.prepare_frozen_embeddings_mask()

# 6. Load truncated models (using only first 3 layers for efficiency)
teacher_model = Gemma3ForAIM.from_pretrained(
    "google/gemma-2-2b",
    n_layers=3,
)
student_model = Gemma3ForAIM.from_pretrained(
    "google/gemma-2-2b",  # Same architecture, different tokenizer
    n_layers=3,
    frozen_embeddings_mask=frozen_mask,
)

# 7. Train embeddings
learned_embeddings = matt.train(teacher_model, student_model)

# 8. Load full model and replace embeddings
full_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.bfloat16,
)
full_model.get_input_embeddings().weight.data.copy_(learned_embeddings)

# 9. Save adapted model
full_model.save_pretrained("output/gemma-2-2b-llama-tokenizer")
student_tokenizer.save_pretrained("output/gemma-2-2b-llama-tokenizer")
```

## Adding New Architectures

Extend `PreTrainedModelForAIM` and implement 4 methods:

```python
from matt.modeling import PreTrainedModelForAIM

class MyModelForAIM(PreTrainedModelForAIM):
    def get_base_model(self, model):
        return model.model  # Return base transformer

    def get_layers(self):
        return self.model.layers  # Return layer list

    def set_layers(self, layers):
        self.model.layers = nn.ModuleList(layers)

    def create_partly_frozen_embeddings(self, frozen_mask):
        return PartlyFrozenEmbeddings(
            embeddings=self.model.embed_tokens,
            frozen_mask=frozen_mask,
        )
```

## Configuration

### Key Configuration Options

```python
config = MATTConfig(
    # Training
    max_steps=10000,           # Total training steps
    batch_size=4,              # Samples per batch
    lr=1e-4,                   # Learning rate
    loss="mse",                # Loss function: "mse" or "cosine"
    aim_impl="aim",            # AIM variant: "aim" or "aim_star"

    # Hardware
    accelerator="gpu",         # "gpu", "cpu", "tpu"
    devices=-1,                # Number of devices (-1 = all)
    precision="bf16-mixed",    # Mixed precision training
    strategy="auto",           # Distributed strategy

    # Optimization
    gradient_clip_val=1.0,     # Gradient clipping
    accumulate_grad_batches=1, # Gradient accumulation

    # Logging & Checkpoints
    log_every_n_steps=50,
    save_every_n_steps=5000,
    use_wandb=False,           # Enable W&B logging
)
```

See [`MATTConfig`](matt/config.py) for all 30+ parameters.

## Advanced Features

### Partial Embedding Freezing

Tokens shared between the teacher and student tokenizers can keep their original embeddings frozen, reducing the number of parameters that need to be learned:

```python
# Identify overlapping tokens using deepfocus
frozen_mask = matt.prepare_frozen_embeddings_mask()

# Only train embeddings for new tokens
student_model = Gemma3ForAIM.from_pretrained(
    model_path,
    n_layers=3,
    frozen_embeddings_mask=frozen_mask,  # Freeze overlapping tokens
)
```

**When to freeze vs. unfreeze:**
- **Single-language transfer** (e.g., extending to Ukrainian): freeze overlapping tokens to preserve the model's existing language performance.
- **Multilingual transfer** (5+ languages simultaneously): unfreeze all embeddings to allow the model the semantic flexibility needed across diverse scripts and languages.

### AIM Implementation Variants

Choose between two AIM implementations:

- **`aim`** (default): For each text segment acting as a query, aligns how much attention it pays to every preceding segment (weighted value states). Captures the full pairwise influence matrix — more accurate, but uses more VRAM and time.
- **`aim_star`**: Only aligns the final output state of each segment (the sum over all attended values), skipping the pairwise breakdown. Faster with slightly lower accuracy.

Both variants apply the objective at the last included layer only, which is both cheaper and better than accumulating the loss across all layers.

```python
config = MATTConfig(aim_impl="aim_star")  # Use simplified version
```

### NTP Loss for Untied Embeddings

Transformer models come in two variants regarding their embedding matrices:

- **Tied embeddings** (e.g., all Gemma models): the input embedding matrix is shared with the LM head. Training the input embeddings automatically trains the output side too, so AIM alone is sufficient.
- **Untied embeddings** (e.g., large Qwen3 variants): the input embeddings and the LM head are separate weight matrices. AIM only trains the input embeddings, leaving the LM head uninitialized for new tokens. Adding a **Next Token Prediction (NTP) loss** ensures the output embeddings are trained as well.

When NTP is enabled, the student model is kept at full depth (not truncated) so that it can produce a proper language modeling loss. The two losses are combined with **dynamic scaling**: at each step, the NTP loss is multiplied by `aim_loss / ntp_loss` so both components stay at the same magnitude, preventing one from dominating the other.

> **Note:** using NTP requires a full forward pass through the student model at every step instead of through only the first ~1/3 of layers, which moderately increases memory usage and training time.

```python
# 1. Enable NTP in the config
config = MATTConfig(
    with_ntp=True,
    max_steps=10000,
    lr=1e-4,
)

# 2. Keep the student model at full depth (n_layers is still used for AIM target layer)
student_model = Qwen3ForAIM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    n_layers=9,        # layer used for AIM signal (~1/3 of 28)
    with_ntp=True,     # keep full model, train output embeddings too
)

# 3. After training, also update the LM head weights
learned_embeddings = matt.train(teacher_model, student_model)

full_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
full_model.get_input_embeddings().weight.data.copy_(learned_embeddings)
# Copy output embeddings from the trained student model
full_model.get_output_embeddings().weight.data.copy_(
    student_model.get_output_embeddings().weight.data
)
```

### Learning Rate Scheduling

By default, MATT uses a constant learning rate with AdamW. For longer runs it can help to add linear warmup and a decay phase at the end (warmup-stable-decay schedule):

```python
config = MATTConfig(
    lr=1e-4,
    num_warmup_steps=1000,   # linear ramp-up from 0 to lr
    num_decay_steps=5000,    # linear decay to 0 at the end
    max_steps=50000,
)
```

Setting either value to `0` (the default) disables that phase.

### Loss Functions

- **MSE** (default): Better for generative tasks; recommended for most use cases.
- **Cosine**: Better for discriminative benchmarks.

```python
config = MATTConfig(loss="cosine")
```

### Distributed Training

MATT uses PyTorch Lightning for seamless distributed training:

```python
config = MATTConfig(
    devices=4,              # Use 4 GPUs
    strategy="ddp",         # Distributed Data Parallel
    accelerator="gpu",
)
```

### Monitoring

Enable Weights & Biases for comprehensive logging:

```python
config = MATTConfig(
    use_wandb=True,
    wandb_project="my-tokenizer-transfer",
)
```

## Examples

See [`examples/`](examples/) for complete working examples:

- [`basic_usage.py`](examples/basic_usage.py): Full end-to-end transfer workflow

## Limitations

- **Untied embeddings require NTP**: Models with separate input and LM head weights (e.g., Qwen3) need `with_ntp=True` to properly train the output embeddings. This requires a full forward pass through the student model at each step instead of only the first ~1/3 of layers, moderately increasing memory and training time. See [NTP Loss for Untied Embeddings](#ntp-loss-for-untied-embeddings).
- **Encoder-only models not tested**: Applying MATT to encoder-only architectures would require removing the causal constraint from the AIM objective.
- **Embedding-only training**: MATT is designed as an efficient warm-up, not a replacement for continued pretraining. Full fine-tuning after MATT is expected to yield further gains, especially for large vocabulary extensions.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MATT in your research, please cite:

```bibtex
@article{haltiuk2025model,
  title={Model-Aware Tokenizer Transfer},
  author={Haltiuk, Mykola and Smywi{\'n}ski-Pohl, Aleksander},
  journal={arXiv e-prints},
  pages={arXiv--2510},
  year={2025}
}
```

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Uses [Transformers](https://huggingface.co/transformers/) for model loading
- Token overlap detection via [deepfocus](https://github.com/Goader/deepfocus)
