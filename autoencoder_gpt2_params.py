import torch
from gpt_utils import get_tokenizer

dataset_name="wikitext"
dataset_config="wikitext-103-v1"
autoencoder_experiment = "Autoencoder GPT2"
learning_rate = 1e-3
num_epochs = 3
batch_size = 32

sparse_dimension_factor = 2
lasso_lambda = 1e-7

num_training_subsets = 2
subsets_max_size = 1
tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
