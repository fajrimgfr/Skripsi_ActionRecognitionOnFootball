import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
import numpy as np
from src import constants
from src.annotations import get_data
from src.datasets import ActionDataset
from src.models.lstm_model import LSTMActionSpotting
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
best_val_loss = float('inf')
save_path = "./best_model.pth"

class_counts = np.array([3187124, 53, 403, 923, 1357, 1179, 3152, 3015, 2115, 17995, 7965, 6763, 3386, 1009, 2497, 1064])
class_weights = 1.0 / (class_counts + 1e-6)  # tambahkan epsilon biar nggak div by zero
class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize supaya rata

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features = features.float().to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        logits = model(features)
        
        logits = logits.view(-1, constants.num_classes)
        labels = labels.view(-1)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train - epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)
            logits = model(features)
            logits_2d = logits.view(-1, constants.num_classes)
            labels_1d = labels.view(-1)
            loss = criterion(logits_2d, labels_1d)
            val_loss += loss.item()

            preds = torch.argmax(logits_2d, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_1d.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Val - epoch {epoch+1}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=constants.classes))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
