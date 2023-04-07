import torch
from tqdm import tqdm
from data.data_processor import create_dataloaders, load_data
from train.model_utils import load_model

MODEL_PATH = "path/to/your/saved/model.pt"

def evaluate_model(model, test_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_iterator = tqdm(test_loader, desc="Testing", total=len(test_loader))
    with torch.no_grad():
        for inputs, attention_mask, labels in test_iterator:
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            logits = model(inputs, attention_mask=attention_mask)

            loss = criterion(logits, labels)

            test_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_iterator.set_postfix({"Test Loss": loss.item()})

    test_loss /= len(test_loader)
    accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data here
    data_dir = "data/aclImdb"
    test_dir = f"{data_dir}/test"
    test_texts, test_labels = load_data(test_dir)

    batch_size = 8
    _, _, test_loader = create_dataloaders([], [], [], [], test_texts, test_labels, batch_size)

    model = load_model(MODEL_PATH, device)

    evaluate_model(model, test_loader, device)