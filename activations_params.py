import torch
from gpt_utils import get_tokenizer

transformer_run_id = "17825e172dbe48b8934c57b59e85d83c"
autoencoder_run_id = "d3b46482c339417fb06029ac60a29976"

num_training_subsets = 2
subsets_max_size = 1
batch_size = 64

tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)