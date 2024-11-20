import torch
from gpt_utils import get_tokenizer

transformer_run_id = "59c595d993e94586b92816aa4ecbac62"
autoencoder_run_id = "c2d69768bb4a465980facdeb9dd96cbb"

num_training_subsets = 2
subsets_max_size = 0.5
batch_size = 64

tokenizer = get_tokenizer("gpt2")
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)