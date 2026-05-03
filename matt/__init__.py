"""
MATT: Model-Aware Tokenizer Transfer

A Python package for transferring tokenizers using Attention Influence Modeling (AIM).
"""
import logging

__version__ = "0.1.0"
__author__ = "Mykola Haltiuk"
__email__ = "kaljandlux@gmail.com"

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import main classes and functions
from .config import MATTConfig
from .matt import MATT, MATTOutput

from .modeling.base import PreTrainedModelForAIM
from .modeling.embeddings import PartlyFrozenEmbeddings
from .modeling.llama import LlamaForAIM
from .modeling.gemma3 import Gemma3ForAIM
from .modeling.qwen3 import Qwen3ForAIM
from .modeling.gpt_neox import GPTNeoXForAIM

__all__ = [
    "MATTConfig",
    "MATT",
    "MATTOutput",
    "PreTrainedModelForAIM",
    "PartlyFrozenEmbeddings",
    "LlamaForAIM",
    "Gemma3ForAIM",
    "Qwen3ForAIM",
    "GPTNeoXForAIM",
]
