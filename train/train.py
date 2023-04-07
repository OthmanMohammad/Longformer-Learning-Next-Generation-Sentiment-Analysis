import torch
from torch import nn, optim
from longformer_model.longformer_for_sequence_classification import LongformerForSequenceClassification
from longformer_model.custom_longformer import CustomLongformer
from longformer_model.longformer_classification_head import LongformerClassificationHead
from data.data_processor import create_dataloaders, load_data
from train.model_utils import load_model, save_model
from transformers import LongformerTokenizer
from tqdm import tqdm


def load_split_data():
    # Load your data here
    data_dir = "data/aclImdb"
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir = f"{data_dir}/test"

    train_texts, train_labels = load_data(train_dir)
    valid_texts, valid_labels = load_data(valid_dir)
    test_texts, test_labels = load_data(test_dir)

    return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels


def train_model(model, train_loader, valid_loader, device, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        train_iterator = tqdm(train_loader, desc="Training", total=len(train_loader))
        for inputs, attention_mask, labels in train_iterator:
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iterator.set_postfix({"Train Loss": loss.item()})

        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        # Validation loop
        valid_iterator = tqdm(valid_loader, desc="Validation", total=len(valid_loader))
        with torch.no_grad():
            for inputs, attention_mask, labels in valid_iterator:
                inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(inputs, attention_mask=attention_mask)

                loss = criterion(outputs.logits, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                valid_iterator.set_postfix({"Valid Loss": loss.item()})

        valid_loss /= len(valid_loader)
        accuracy = correct / total

        print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}")

        save_model(model, f"longformer_sentiment_epoch_{epoch + 1}.pth")
