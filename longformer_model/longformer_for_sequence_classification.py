import torch.nn as nn
from .custom_longformer import CustomLongformer
from .longformer_classification_head import LongformerClassificationHead

class LongformerForSequenceClassification(nn.Module):
    def __init__(self, longformer, classification_head):
        super(LongformerForSequenceClassification, self).__init__()
        self.longformer = longformer
        self.classification_head = classification_head

    def forward(self, x, attention_mask=None):
        x = self.longformer(x, attention_mask=attention_mask)
        x = self.classification_head(x)
        return x
