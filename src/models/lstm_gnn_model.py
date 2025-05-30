import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LSTM_GNN_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, gnn_hidden_dim, num_classes, num_layers=1):
        super(LSTM_GNN_Model, self).__init__()
        
        # LSTM layer bidirectional
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Graph Conv layers
        self.gcn1 = GCNConv(lstm_hidden_dim * 2, gnn_hidden_dim)
        self.gcn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        
        # Classifier
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # LSTM output: [batch_size, seq_len, lstm_hidden_dim*2]
        lstm_out, _ = self.lstm(x)
        
        # Flatten batch dan seq_len untuk graph input: [batch_size*seq_len, lstm_hidden_dim*2]
        lstm_out_reshape = lstm_out.view(batch_size * seq_len, -1)
        
        # Graph conv layers (dengan edge_index)
        gcn_out = F.relu(self.gcn1(lstm_out_reshape, edge_index))
        gcn_out = F.relu(self.gcn2(gcn_out, edge_index))
        
        # Classifier output: [batch_size*seq_len, num_classes]
        logits = self.classifier(gcn_out)
        
        # Kembalikan ke bentuk [batch_size, seq_len, num_classes]
        logits = logits.view(batch_size, seq_len, -1)
        return logits