import os
import numpy as np
import torch
from torch.nn.functional import softmax
from src.models.lstm_model import LSTMActionSpotting
from src import constants
from configs import action
from src.annotations import get_data
from tqdm import tqdm
import re
import json

device = torch.device(action.device)
model = LSTMActionSpotting().to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

save_dir = "./data/experiments/predictions/lstm_3"
os.makedirs(save_dir, exist_ok=True)

test_games = get_data(constants.test_games)

for half in tqdm(test_games, desc=f"Predicting events..."):
    data_path = half["data_path"]
    data = np.load(data_path)  # shape: (num_frames, feat_dim)
    game_id = f"{half['half']}_{half['game']}"

    predictions = {}
    with torch.no_grad():
        for i in range(len(data)):
            frame_feat = data[i]  # ambil satu frame saja
            frame_tensor = torch.tensor(frame_feat).unsqueeze(0).float().to(device)  # shape: (1, feat_dim)

            logits = model(frame_tensor)
            probs = softmax(logits, dim=1).squeeze()
            pred_class = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

            if pred_class != 0 and confidence > 0.7:
                predictions[i] = {
                    "label": constants.classes[pred_class],
                    "confidence": round(confidence, 3)
                }

    print(f"Done predicting {game_id}, found {len(predictions)} events")

    # Save prediction JSON
    file_safe_game_id = re.sub(r'[\\/*?:"<>|]', '_', game_id)
    with open(os.path.join(save_dir, f"{file_safe_game_id}_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {file_safe_game_id}_predictions.json")
