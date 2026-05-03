"""
Basic usage example for Model-Aware Tokenizer Transfer (MATT).
"""
import os
import gc

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from matt import MATT, MATTConfig
from matt.modeling.gemma3 import Gemma3ForAIM


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    """Demonstrate basic usage of MATT."""

    teacher_model_path = 'google/gemma-3-4b-pt'
    student_model_path = 'Goader/gemma-3-4b-pt-focus'
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)

    matt = MATT(
        config=MATTConfig(
            use_wandb=True,
            wandb_project='matt',
            aim_impl='aim',
            loss='mse',
            lr=1e-4,
            save_every_n_steps=5000,
            # ...
        ),
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
    )

    # We use a small dataset for demonstration purposes.
    dataset = load_dataset('Goader/hplt-uk-100k', split='train')

    matt.prepare_dataset(
        dataset=dataset,
        max_length=256,
        force_reprocess=False,
        batch_size=1000,
        num_proc=os.cpu_count(),
    )

    # If you do not want to freeze embeddings, you can skip this step
    # and pass None to the student model constructor.
    frozen_embeddings_mask = matt.prepare_frozen_embeddings_mask()

    # Higher n_layers is advisable for better performance.
    n_layers = 3
    teacher_model = Gemma3ForAIM.from_pretrained(
        teacher_model_path,
        n_layers=n_layers,
    )
    student_model = Gemma3ForAIM.from_pretrained(
        student_model_path,
        n_layers=n_layers,
        frozen_embeddings_mask=frozen_embeddings_mask,
    )

    embeddings = matt.train(teacher_model, student_model)

    # Clearing memory
    del teacher_model, student_model
    clear_memory()

    # Substituting embeddings
    model = AutoModelForCausalLM.from_pretrained(student_model_path, torch_dtype=torch.bfloat16)
    model.get_input_embeddings().weight.data.copy_(embeddings)
    model.save_pretrained('output/models/gemma-3-4b-pt-matt')
