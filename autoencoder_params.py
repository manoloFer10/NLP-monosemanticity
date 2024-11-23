import torch
from gpt_utils import get_tokenizer

autoencoder_experiment = "Autoencoder"

transformer_run_id = "1631cdf63904427fb5833afa9372b625"

dataset_name="wikitext"
dataset_config="wikitext-103-v1"
learning_rate = 1e-4
num_epochs = 1
batch_size = 64

sparse_dimension_factor = 16
lasso_lambda = 1e-4
activations_threshold = 1e-5

subsets_max_size = 20
num_training_subsets = 30
tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
