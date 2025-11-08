from transformers import GPT2Tokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

def get_data(split):
    with open(f'./data/{split}.txt') as f:
        return f.readlines()

class TokenizedData(ABC):
    @property
    @abstractmethod
    def vocab_size(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def padding_token(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataloaders(self):
        raise NotImplementedError()

class GPTTokenizedData(TokenizedData):
    def __init__(self, batch_size=64):
        self._prepare_tokenizer()
        self.batch_size = batch_size
        self._prepare_dataloaders()

    def _prepare_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def padding_token(self):
        return self.tokenizer.pad_token
    
    @property
    def dataloaders(self):
        return self._dataloaders

    def _prepare_dataloaders(self):
        def tokenize_with_eos(samples, tokenizer):
            result = {'input_ids': [], 'attention_mask': []}

            for sent in samples:
                # using this tokenization the longest sample is 143 tokens
                tokens = tokenizer.encode(sent, truncation=True, max_length=150)

                tokens_with_eos = [tokenizer.eos_token_id] + tokens + [tokenizer.eos_token_id]

                result['input_ids'].append(tokens_with_eos)
                result['attention_mask'].append([1] * len(tokens_with_eos))

            return result    
        
        self._dataloaders = {}
        for split in ['train', 'test', 'val']:
            data = get_data(split)
            
            encoded_input = tokenize_with_eos(data, self.tokenizer)
            tokenized_dataset = Dataset.from_dict(encoded_input)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            self._dataloaders[split] = DataLoader(
                tokenized_dataset,
                batch_size=self.batch_size,
                collate_fn=data_collator,
            )
