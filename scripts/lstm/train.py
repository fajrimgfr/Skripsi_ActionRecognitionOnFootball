import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from src import constants
from src.annotations import get_data, get_videos_sampling_weights
from src.datasets import TrainActionDataset, ValActionDataset
from src.sliding_window_datasets import SlidingWindowDataset
from src.sampler import BalancedFrameSampler
from src.models.lstm_model import LSTMActionSpotting
from src.indexes import StackIndexesGenerator, FrameIndexShaker
from src.target import MaxWindowTargetsProcessor
from configs import action
from tqdm import tqdm

batch_size = action.batch_size
num_epochs = action.num_epochs
learning_rate = action.learning_rate

train_games = get_data(constants.train_games)
val_games = get_data(constants.valid_games)

indexes_generator = StackIndexesGenerator(
    action.sequence,
    action.step,
)
frame_index_shaker = FrameIndexShaker(action.shifts, action.weights, action.prob)

videos_sampling_weights = get_videos_sampling_weights(
    # train_games, action.action_window_size, action.action_prob,  action.action_weights
    train_games, action.action_window_size, action.action_prob
)

targets_processor = MaxWindowTargetsProcessor(
        window_size=action.sequence
    )

train_dataset = TrainActionDataset(train_games, constants.classes, indexes_generator, videos_sampling_weights, targets_processor, frame_index_shaker)
val_dataset = ValActionDataset(val_games, constants.classes, indexes_generator, videos_sampling_weights, targets_processor, frame_index_shaker)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

device = torch.device(action.device)
model = LSTMActionSpotting().to(device)

resume_training = False
checkpoint_path = "./best_model.pth"

optimizer = Adam(model.parameters(), lr=learning_rate)
best_val_loss = float('inf')
save_path = "./best_model.pth"

criterion = nn.BCEWithLogitsLoss()

if resume_training and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Loaded pretrained model from checkpoint.")

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
            logits = model(features)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }, save_path)
        print(f"Saved best model at epoch {epoch+1} with val loss {best_val_loss:.4f}")
