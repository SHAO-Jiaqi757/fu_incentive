import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = list(data)  # Convert iterator to list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = [item[0] - 1 for item in self.data]  # AG News labels start from 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]  # AG News format is (label, text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label - 1, dtype=torch.long)  # AG News labels start from 1
        }



class BackdooredTextDataset(TextDataset):
    def __init__(self, data, tokenizer, max_length=128, backdoor_pattern=None, backdoor_label=None):
        self.data = list(data)  # Convert iterator to list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = [item[0] - 1 for item in self.data]  # AG News labels start from 1
        self.backdoor_pattern = backdoor_pattern
        self.backdoor_label = backdoor_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]  # AG News format is (label, text)
        
        # Apply backdoor if pattern is set
        if self.backdoor_pattern:
            text = self.backdoor_pattern + " " + text
            if self.backdoor_label is not None:
                label = self.backdoor_label

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label - 1, dtype=torch.long)  # AG News labels start from 1
        }
