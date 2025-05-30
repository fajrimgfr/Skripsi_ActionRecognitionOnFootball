import torch.nn as nn

class LSTMActionSpotting(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # because bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)  # out: [batch, seq, hidden*2]
        logits = self.classifier(out)  # [batch, seq, num_classes]
        return logits
