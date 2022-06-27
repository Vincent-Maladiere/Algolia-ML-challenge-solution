import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

import nltk
nltk.download('stopwords') # 2,400 stopwords for 11 languages
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


class BertEmbedder:
    
    def __init__(self, layers=None):
        self.layers = [-4, -3, -2, -1] if layers is None else layers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    def get_embeddings(self, text):
        
        # 0. preprocess: remove stop words
        text = " ".join([token for token in text.split() if token.lower() not in stop_words])

        # 1. tokenize
        encoded = self.tokenizer.encode_plus(text, return_tensors="pt")
        
        # 2. generate hidden states
        with torch.no_grad():
            output = self.model(**encoded)
        states = output.hidden_states
        output = torch.stack([states[i] for i in self.layers]).sum(0).squeeze()
        
        # 3. select output linked to token and average them
        tokens = text.split()
        sentence_embeddings = []
        for token in tokens:
            idx = tokens.index(token)
            token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

            # Only select the tokens that constitute the requested sentence to remove special tokens
            word_tokens_output = output[token_ids_word]
            word_tokens_output = word_tokens_output.mean(dim=0)
            sentence_embeddings.append(word_tokens_output)

        if not sentence_embeddings:
            print(f" # [BertEmbedder] error from input {text}, skipped")
            return None

        sentence_embeddings = torch.stack(sentence_embeddings)
        
        return sentence_embeddings.mean(dim=0)