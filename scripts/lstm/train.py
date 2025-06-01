import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, average_precision_score
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

# 🔥 Perhitungan pos_weight (jika kamu tahu distribusi class imbalance)
pos_weight = torch.ones([15]).to(device)  # <-- ganti kalau ada distribusi spesifik
criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = Adam(model.parameters(), lr=learning_rate)
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
    print(f"Train - epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.float().to(device), labels.float().to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()  # probs shape: [batch, seq, num_classes]
            preds = (probs > 0.5).astype(np.float32)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_probs = np.vstack(all_probs).reshape(-1, probs.shape[-1])
    all_preds = np.vstack(all_preds).reshape(-1, probs.shape[-1])
    all_labels = np.vstack(all_labels).reshape(-1, probs.shape[-1])

    # Average precision pakai probabilitas, bukan binary preds
    avg_precision_overall = average_precision_score(all_labels, all_probs, average='macro')
    avg_precision_per_class = average_precision_score(all_labels, all_probs, average=None)

    # Binary accuracy overall
    binary_accuracy_overall = (all_preds == all_labels).mean()

    # Binary accuracy per class
    binary_accuracy_per_class = []
    for i in range(all_labels.shape[1]):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        binary_accuracy_per_class.append(acc)

    print(f"Val - epoch {epoch+1}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Average Precision (Overall): {avg_precision_overall:.4f}")
    print(f"Average Precision per Class: {avg_precision_per_class}")
    print(f"Binary Accuracy (Overall): {binary_accuracy_overall:.4f}")
    print(f"Binary Accuracy per Class: {binary_accuracy_per_class}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
