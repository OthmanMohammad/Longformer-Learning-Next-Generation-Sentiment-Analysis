# Longformer Sentiment Analysis

This project is an implementation of the Longformer model for sentiment analysis, particularly on the IMDB movie review dataset. Longformer, introduced in the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan, is designed to handle long documents by employing a combination of sliding-window and global attention mechanisms.

The project structure is as follows:



```
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── data_processor.py
├── evaluate/
│   ├── __init__.py
│   └── evaluate.py
├── longformer_model/
│   ├── __init__.py
│   ├── custom_longformer.py
│   ├── longformer_classification_head.py
│   ├── longformer_for_sequence_classification.py
│   ├── longformer_layer.py
│   └── longformer_self_attention.py
├── predict/
│   ├── __init__.py
│   └── predict.py
├── train/
│   ├── __init__.py
│   ├── model_utils.py
│   └── train.py
├── .gitignore
├── LICENSE
├── __init__.py
├── download_dataset.py
├── main.py
└── requirements.txt
```


## Contents

1. [Components](#components)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Longformer Model](#longformer-model)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Prediction](#prediction)
8. [License](#license)
9. [Author](#author)


## Components

- **Longformer Self-Attention**: The self-attention mechanism is implemented in the LongformerSelfAttention class within the longformer_self_attention.py module. Although it doesn't include sliding window and global attention mechanisms explicitly, it is a base for the standard self-attention mechanism.
  - `longformer_self_attention.py`: Contains the implementation of the Longformer self-attention mechanism, with the query, key, and value projections.

```python
class LongformerSelfAttention(nn.Module):
```

- **Longformer Layer**: The LongformerLayer class within the `longformer_layer.py`: Implements a single transformer layer in the Longformer model, including the self-attention and feed-forward sublayers.

```python
class LongformerLayer(nn.Module):
```

- **Longformer Model**: The CustomLongformer class within the `custom_longformer.py` module assembles the Longformer model, consisting of embedding layers and a sequence of Longformer layers.
  - `custom_longformer.py`: Contains the implementation of the Longformer model, including the input and positional embeddings and the layer structure.

```python
class CustomLongformer(nn.Module):
```

- **Classification Head**: The LongformerClassificationHead class within the `longformer_classification_head.py` module defines the classification head for the sentiment analysis task.

```python
class LongformerClassificationHead(nn.Module):
```

- **Longformer for Sequence Classification**: The LongformerForSequenceClassification class within the `longformer_for_sequence_classification.py` module combines the Longformer model and classification head into a single module for sequence classification.

```python
class LongformerForSequenceClassification(nn.Module):
```


## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```


## Dataset

The IMDB movie review dataset consists of 50,000 movie reviews, labeled as either positive or negative. The dataset is available at [this link](http://ai.stanford.edu/~amaas/data/sentiment/). To download and preprocess the dataset, run:

```bash
python download_dataset.py
```


This will download and extract the dataset into the `data/aclImdb` directory.

The dataset is split into three parts: training, validation, and testing. The `data/data_processor.py` module contains the necessary functions to load and preprocess the dataset.

## Longformer Model

The Longformer model is implemented in the `longformer_model` directory. The main components of the model include:

- `custom_longformer.py`: Contains the implementation of the Longformer model, including the self-attention mechanism, positional embeddings, and layer structure.
- `longformer_classification_head.py`: Defines the classification head for the sentiment analysis task.
- `longformer_for_sequence_classification.py`: Combines the Longformer model and classification head into a single module for sequence classification.
- `longformer_layer.py`: Implements a single transformer layer in the Longformer model.
- `longformer_self_attention.py`: Implements the sliding window and global attention mechanisms of the Longformer model.

## Training

The training process is defined in the train/train.py module. The main function train_model() takes a pre-defined Longformer model, data loaders for training and validation sets, and other hyperparameters to train the model.

Please note that training a Longformer model can be computationally intensive and might require significant computational resources. For efficient training, it is recommended to use powerful machines, such as AWS EC2 instances with at least an `r5.16xlarge` configuration or GPU-enabled instances like `g4dn.metal`.

```bash
python main.py
```

If you face issues with training due to limited resources, you can adjust the model's configuration by reducing the number of layers, attention heads, or hidden size. However, please note that decreasing these configurations may lead to reduced performance. You can make these adjustments in the `main.py` file where the CustomLongformer class is instantiated and update the hyperparameters accordingly.

## Evaluation

The evaluation process is implemented in the `evaluate/evaluate.py` module. This module provides a function to compute the accuracy of a trained model on the test dataset. To perform evaluation, import the evaluate_model function from the `evaluate/evaluate.py` module and provide the trained model and test data loader as arguments.

```python
from evaluate.evaluate import evaluate_model

# Load the trained model and test data loader
# ...

accuracy = evaluate_model(model, test_loader, device)
print(f"Test accuracy: {accuracy:.2f}")
```

## Prediction

The prediction process is implemented in the `predict/predict.py` module. This module provides a function called predict_sentiment that takes a trained model, tokenizer, and a text input to predict the sentiment of the input text.

```python
from predict.predict import predict_sentiment

# Load the trained model and tokenizer
# ...

text = "I found the characters unrelatable and the script poorly written, which made it difficult for me to enjoy the movie."
sentiment = predict_sentiment(model, tokenizer, text, device)
print(f"Predicted sentiment: {sentiment}")
```

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as you wish, but please maintain the original author's attribution.

## Author

**Mohammad Othman**\
[Github](https://github.com/OthmanMohammad)
