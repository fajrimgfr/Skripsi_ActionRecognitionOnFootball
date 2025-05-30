import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.lstm_gnn_model import LSTM_GNN_Model
from dataset import SoccerDataset
from utils.graph_utils import create_chain_edge_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 512
lstm_hidden_dim = 256
gnn_hidden_dim = 128
num_classes = 3  # misal: no-event, goal, foul (ubah sesuai data)
batch_size = 8
seq_len = 100  # contoh, sesuaikan dengan dataset
num_epochs = 10
learning_rate = 0.001

# Load dataset
train_dataset = SoccerDataset("data/train_features.npy", "data/train_labels.npy")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model init
model = LSTM_GNN_Model(input_dim, lstm_hidden_dim, gnn_hidden_dim, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Buat edge_index graph untuk batch
        edge_index = create_chain_edge_index(batch_size=x.size(0), seq_len=x.size(1), device=device)
        
        optimizer.zero_grad()
        output = model(x, edge_index)  # output shape: [batch_size, seq_len, num_classes]
        
        # CrossEntropyLoss expects [N, C] and labels [N], kita flatten dulu
        loss = criterion(output.view(-1, num_classes), y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")