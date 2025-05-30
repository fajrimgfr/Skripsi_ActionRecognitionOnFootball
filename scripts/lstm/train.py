import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score, accuracy_score
from src import constants
from src.annotations import get_data
from src.datasets import ActionDataset
from src.models.lstm_model import LSTMActionSpotting
import os
from configs import action

batch_size = action.batch_size
num_epochs = action.num_epochs
learning_rate = action.learning_rate

train_games = get_data(constants.train_games)
val_games = get_data(constants.valid_games)

train_dataset = ActionDataset(train_games)
val_dataset = ActionDataset(val_games)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device(action.device)
model = LSTMActionSpotting().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = BCEWithLogitsLoss()

best_val_loss = float('inf')
save_path = "./best_model.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.float().to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.float().to(device), labels.float().to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    accuracy = (all_preds == all_labels).mean()

    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val F1 Score: {f1:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
