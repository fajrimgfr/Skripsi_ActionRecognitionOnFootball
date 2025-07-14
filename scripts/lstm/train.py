import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src import constants
from src.annotations import get_data
from src.datasets import ActionDataset
from src.sliding_window_datasets import SlidingWindowDataset
from src.sampler import BalancedFrameSampler
from src.models.lstm_model import LSTMActionSpotting
from configs import action
from tqdm import tqdm

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

criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        features = features.float().to(device)
        labels = labels.float().to(device)

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
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            features = features.float().to(device)
            labels = labels.float().to(device)
            logits_2d = model(features)
            loss = criterion(logits_2d, labels_1d)
            val_loss += loss.item()

            probs = torch.sigmoid(logits_2d)
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(f"Val - epoch {epoch+1}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(classification_report(all_labels, all_preds, target_names=constants.classes))
    print("F1 Score (macro):", f1_score(all_labels, all_preds, average='macro'))
    print("F1 Score (samples):", f1_score(all_labels, all_preds, average='samples'))
    print("mAP (macro):", average_precision_score(all_labels, all_preds, average='macro'))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
