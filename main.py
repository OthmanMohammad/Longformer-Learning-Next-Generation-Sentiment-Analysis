from train.train import train_model, load_split_data
import torch
from torch import nn, optim
from data.data_processor import create_dataloaders, load_data
from transformers import LongformerTokenizer
from longformer_model.custom_longformer import CustomLongformer
from longformer_model.longformer_classification_head import LongformerClassificationHead
from longformer_model.longformer_for_sequence_classification import LongformerForSequenceClassification

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = load_split_data()
    batch_size = 8
    train_loader, valid_loader, test_loader = create_dataloaders(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, batch_size)

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    vocab_size = tokenizer.vocab_size
    max_position_embeddings = 4096
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    intermediate_size = 3072
    dropout_rate = 0.1
    num_labels = 2

    longformer = CustomLongformer(vocab_size, max_position_embeddings, hidden_size, num_layers, num_heads, intermediate_size, dropout_rate)
    classification_head = LongformerClassificationHead(hidden_size, num_labels)
    model = LongformerForSequenceClassification(longformer, classification_head)

    epochs = 3
    learning_rate = 1e-5

    train_model(model, train_loader, valid_loader, device, epochs, learning_rate)
