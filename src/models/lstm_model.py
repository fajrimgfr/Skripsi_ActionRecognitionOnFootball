import torch.nn as nn
from src import constants

class LSTMActionSpotting(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=constants.num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return logits
