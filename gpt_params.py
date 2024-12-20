import math
import torch
from gpt_utils import get_tokenizer

transformer_experiment = "Transformer"

dataset_name="wikitext"
dataset_config="wikitext-103-v1"
subsets_max_size = 1
num_training_subsets = 2

tokenizer = get_tokenizer("gpt2")
vocab_size = tokenizer.vocab_size

context_length = 40
embedding_dim = 128
num_of_attention_heads = 8
num_of_blocks = 1

batch_size = 16
learning_rate = 0.001
dropout = 0

eval_interval = 20
epochs = 1

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)