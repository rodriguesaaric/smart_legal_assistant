import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from torch.optim import AdamW
import os
from dataset import prepare_dataloaders

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, inputs, targets):
    
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def train_model(train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = AutoModelForTokenClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased", 
        num_labels=2 
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 2)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            # Apply Focal Loss
            loss = criterion(active_logits, active_labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
    output_dir = "./saved_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Training complete. Model weights saved to {output_dir}")

if __name__ == "__main__":
    print("Initializing training sequence...")
    train_loader,val_loader = prepare_dataloaders(json_path="train.json", batch_size=16)
    train_model(train_loader,val_loader)