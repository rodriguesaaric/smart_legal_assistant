import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class LegalClauseDataset(Dataset):
    
    def __init__(self, data, tokenizer_name="nlpaueb/legal-bert-base-uncased", max_length=512):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        clause_starts = item.get("clause_starts", [])

        # Tokenize the document
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True, 
            return_tensors="pt"
        )

        labels = torch.zeros(self.max_length, dtype=torch.long)
        offsets = encoding.pop("offset_mapping")[0] 
        for start_char in clause_starts:
            for token_idx, (token_start, token_end) in enumerate(offsets):
                if token_start == start_char:
                    labels[token_idx] = 1 
                    break

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

def prepare_dataloaders(json_path="training/train.json", batch_size=16):
    
    with open("training/train.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    train_data, val_data = train_test_split(raw_data, test_size=0.20, random_state=42)
    print(f"Dataset Split -> Training: {len(train_data)} docs | Validation: {len(val_data)} docs")

    train_dataset = LegalClauseDataset(train_data)
    val_dataset = LegalClauseDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

