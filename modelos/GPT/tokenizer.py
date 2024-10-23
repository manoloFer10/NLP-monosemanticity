#%%
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers import Encoding

# Define the tokenizer model (e.g., BPE)
tokenizer = Tokenizer(models.BPE())
# Use a pre-tokenizer to split text into basic units (e.g., whitespace splitting)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# Use a decoder to convert tokens back to text (e.g., BPE decoder)
tokenizer.decoder = decoders.BPEDecoder()
# Define the trainer with special tokens if needed
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
# Read your corpus
with open("data/input.txt", "r") as file:
    lines = file.readlines()
# Train the tokenizer on your corpus
tokenizer.train_from_iterator(lines, trainer=trainer)
# Save the tokenizer
tokenizer.save("custom_tokenizer.json")

# %%
