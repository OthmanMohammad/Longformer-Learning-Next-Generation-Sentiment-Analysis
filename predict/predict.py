import torch
from transformers import LongformerTokenizer
from train.model_utils import load_model

MODEL_PATH = "path/to/your/saved/model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

def prepare_sentences(sentences, tokenizer):
    encodings = tokenizer(sentences, truncation=True, padding=True)
    input_ids = torch.tensor(encodings["input_ids"])
    attention_mask = torch.tensor(encodings["attention_mask"])
    return input_ids, attention_mask

def predict_sentences(sentences, model, tokenizer):
    model.eval()
    input_ids, attention_mask = prepare_sentences(sentences, tokenizer)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
    return predictions, probabilities.cpu().numpy()

if __name__ == "__main__":
    model = load_model(MODEL_PATH, device)

    # Multiple sentences
    sentences = [
        "The visual effects in the movie were breathtaking, and the storyline kept me engaged throughout the entire film.",
        "I found the characters unrelatable and the script poorly written, which made it difficult for me to enjoy the movie.",
        "The film's convoluted plot and subpar acting left me feeling unsatisfied and frustrated by the end."
    ]
    predictions, probabilities = predict_sentences(sentences, model, tokenizer)

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sentence {i+1}: {sentences[i]}")
        print(f"Prediction: {pred}, Probability: {prob}")
        print()
