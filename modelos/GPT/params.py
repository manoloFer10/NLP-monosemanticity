import math
import torch
from utils import get_tokenizer

dataset = "wikitext-103-v1"
subsets_max_size = 20
num_training_subsets = 1

tokenizer = get_tokenizer("gpt2")
vocab_size = tokenizer.vocab_size

context_length = 20
embedding_dim = 128
num_of_attention_heads = 8
num_of_blocks = 1

batch_size = 64
learning_rate = 0.0001
dropout = 0.1

eval_interval = 20
epochs = 1

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
