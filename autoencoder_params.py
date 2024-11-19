import torch
from gpt_utils import get_tokenizer

autoencoder_experiment = "Autoencoder"

transformer_run_id = "59c595d993e94586b92816aa4ecbac62"

learning_rate = 1e-3
num_epochs = 3
batch_size = 64

sparse_dimension_factor = 4
lasso_lambda = 1e-7

num_training_subsets = 2
subsets_max_size = 1
tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
