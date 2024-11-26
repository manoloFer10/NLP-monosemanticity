import torch
from gpt_utils import get_tokenizer

transformer_run_id = "1631cdf63904427fb5833afa9372b625"
autoencoder_run_id = "ac16c81835cf43dbb74160fb3caca031"

dataset_name="wikitext"
dataset_config="wikitext-103-v1"
num_training_subsets = 500
subsets_max_size = 1
batch_size = 2048
activations_threshold = 1e-5
window_size=20

tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)