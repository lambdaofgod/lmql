import ipdb
import json
import os
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer, AddedToken
from pathlib import Path as P


class CustomPreTrainedTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path):
        p = P(path)
        tok = Tokenizer.from_file(str(p / "20B_tokenizer.json"))
        return CustomPreTrainedTokenizer(tokenizer=tok)

    @property
    def vocab_size(self):
        return len(self.tokenizer.get_vocab())

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def bos_token_id(self):
        return 209

    @property
    def eos_token_id(self):
        return 50276

    def _tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_token(index)

    def decode(
        self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
    ):
        return self.tokenizer.decode(token_ids, skip_special_tokens)

    def add_special_tokens(self, special_tokens_dict):
        added_tokens = []
        for token, token_id in special_tokens_dict.items():
            added_token = AddedToken(token, single_word=True, lstrip=True)
            self.tokenizer.add_special_token(added_token, token_id)
            added_tokens.append(token)
        return len(added_tokens)

    def num_special_tokens_to_add(self, *args, **kwargs):
        return 0


if __name__ == "__main__":
    # Usage example
    # Initialize your tokenizers.Tokenizer instance here
    p = "/home/kuba/models/rwkv_small/20B_tokenizer.json"
    my_tokenizer = Tokenizer.from_file(p)
    custom_pretrained_tokenizer = CustomPreTrainedTokenizer(my_tokenizer)

    tokens = custom_pretrained_tokenizer.tokenize("Hello, world!")
    print(tokens)
    token_ids = custom_pretrained_tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
    decoded_text = custom_pretrained_tokenizer.decode(token_ids)
    print(decoded_text)

    custom_tokenizer_path = P(p).parent

    # Save the underlying tokenizers.Tokenizer instance
    my_tokenizer.save(os.path.join(custom_tokenizer_path, "tokenizer.json"))

    # Create a tokenizer_config.json file
    tokenizer_config = {
        "tokenizer_class": "CustomPreTrainedTokenizer",
    }

    with open(os.path.join(custom_tokenizer_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    print(custom_pretrained_tokenizer)
