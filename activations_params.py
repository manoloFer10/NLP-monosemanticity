import torch
from gpt_utils import get_tokenizer

transformer_run_id = "1631cdf63904427fb5833afa9372b625"
autoencoder_run_id = "59895464989d4ab488bf5d1cd9d77c9f"

dataset_name="wikitext"
dataset_config="wikitext-103-v1"
num_training_subsets = 2
subsets_max_size = 1
batch_size = 64

tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)