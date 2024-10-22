from transformers import GPT2Tokenizer
from transformers import MBartTokenizer
from transformers import XLMRobertaTokenizer
from transformers import DistilBertTokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif tokenizer_name == "mbart":
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
    elif tokenizer_name == "xlm-roberta":
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    elif tokenizer_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    
    return tokenizer