import torch
from transformers import LongformerForSequenceClassification

def load_model(model_class, model_name, device):
    model = model_class.from_pretrained(model_name)
    return model.to(device)

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).sum().float() / labels.size(0)
    return accuracy.item()
