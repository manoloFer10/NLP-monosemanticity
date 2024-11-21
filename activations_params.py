import torch
from gpt_utils import get_tokenizer

transformer_run_id = "1631cdf63904427fb5833afa9372b625"
autoencoder_run_id = "a7af8e64d7794632b157e5f431b9a17d"

dataset_name="wikitext"
dataset_config="wikitext-103-v1"
num_training_subsets = 2
subsets_max_size = 1
batch_size = 64

tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)