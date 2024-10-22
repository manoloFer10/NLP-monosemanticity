import torch
from utils import get_tokenizer

tokenizer = get_tokenizer("gpt2")
vocab_size = tokenizer.vocab_size

context_length = 20
embedding_dim = 128
num_of_attention_heads = 2
num_of_blocks = 1

batch_size = 32
learning_rate = 0.01
dropout = 0.1

eval_interval = 20
epochs = 3

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)