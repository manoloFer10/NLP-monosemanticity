import torch
from gpt_utils import get_tokenizer

transformer_run_id = "28adb2ef182940b0ac297c6a12e0dba0"
autoencoder_run_id = "b164926a9e5a4190a8108b0d90e59d95"

num_training_subsets = 4
subsets_max_size = 1
batch_size = 64

tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)