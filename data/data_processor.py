import os
from transformers import LongformerTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_data(data_dir):
    texts, labels = [], []
    for label in ["pos", "neg"]:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(0 if label == "neg" else 1)
    return texts, labels

def create_dataloaders(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, batch_size):
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = TensorDataset(torch.tensor(train_encodings["input_ids"]), torch.tensor(train_encodings["attention_mask"]), torch.tensor(train_labels))
    valid_dataset = TensorDataset(torch.tensor(valid_encodings["input_ids"]), torch.tensor(valid_encodings["attention_mask"]), torch.tensor(valid_labels))
    test_dataset = TensorDataset(torch.tensor(test_encodings["input_ids"]), torch.tensor(test_encodings["attention_mask"]), torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
