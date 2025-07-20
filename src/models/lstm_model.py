import torch
import torch.nn as nn
from src import constants

class LSTMActionSpotting(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=constants.num_classes, num_layers=2, dropout=0.2):
        super(LSTMActionSpotting, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)              # x: [B, T, D]
        last_hidden = hn[-1]                      # Ambil output dari layer terakhir
        logits = self.fc(last_hidden)             # [B, num_classes]
        return logits
