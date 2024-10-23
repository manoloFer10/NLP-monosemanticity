import torch
from utils import get_tokenizer


tokenizer = get_tokenizer("gpt2")
vocab_size = tokenizer.vocab_size
# tokenizer = Tokenizer.from_file("custom_tokenizer.json")
# vocab_size = tokenizer.get_vocab_size()

context_length = 20
embedding_dim = 128
num_of_attention_heads = 8
num_of_blocks = 1

batch_size = 64
learning_rate = 0.0001
dropout = 0.1

eval_interval = 20
epochs = 3

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
