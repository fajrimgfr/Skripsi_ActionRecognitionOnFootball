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

context = action.context
device = torch.device(action.device)

model = LSTMActionSpotting().to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

save_dir = "./data/experiments/predictions/lstm_1"
os.makedirs(save_dir, exist_ok=True)

test_games = get_data(constants.test_games)

for half in tqdm(test_games, desc=f"Predicting events..."):
    data_path = half["data_path"]
    data = np.load(data_path)
    game_id = f"{half['half']}_{half['game']}"

    predictions = {}
    with torch.no_grad():
        for i in range(len(data)):
            start = i - context
            end = i + context + 1
            window = np.zeros((2 * context + 1, data.shape[1]), dtype=np.float32)

            for j, f in enumerate(range(start, end)):
                if 0 <= f < len(data):
                    window[j] = data[f]

            window_tensor = torch.tensor(window).unsqueeze(0).float().to(device)
            logits = model(window_tensor)
            # logits = logits[:, 9, :]
            probs = softmax(logits, dim=1).squeeze()
            pred_class = torch.argmax(probs).item()
            confidence = torch.max(probs).item()

            if pred_class != 0 and confidence > 0.7:
                predictions[i] = {
                    "label": constants.classes[pred_class],
                    "confidence": round(confidence, 3)
                }

    print(f"Done predicting {game_id}, found {len(predictions)} events")

    import json
    with open(os.path.join(save_dir, f"{re.sub(r'[\\/*?:"<>|]', '_', game_id)}_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved current predictions")
