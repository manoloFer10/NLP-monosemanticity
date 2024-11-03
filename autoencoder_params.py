import torch
from gpt_utils import get_tokenizer

transformer_run_id = "1c153f8424004a56b805975bd3c45d52"

learning_rate = 1e-3
num_epochs = 3
batch_size = 64

num_training_subsets = 2
subsets_max_size = 1
tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
