import torch
from gpt_utils import get_tokenizer

transformer_experiment = "Training Transformer"
transformer_run_id = "bcf437c7ed984c8d9c3585f7eceb744d"

learning_rate = 1e-3
num_epochs = 3
batch_size = 64

num_training_subsets = 2
subsets_max_size = 1
tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
