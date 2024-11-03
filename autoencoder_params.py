import torch
from gpt_utils import get_tokenizer

transformer_run_id = "28adb2ef182940b0ac297c6a12e0dba0"

learning_rate = 1e-3
num_epochs = 3
batch_size = 64

sparse_dimension_factor = 4

num_training_subsets = 2
subsets_max_size = 1
tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
