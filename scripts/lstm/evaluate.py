import os
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from src.annotations import get_data
from src import constants
from configs import action

test_games = get_data(constants.test_games)
pred_dir = "./data/experiments/predictions/lstm_3"
sequence = action.sequence

all_labels = []
all_preds = []

for game in test_games:
    game_id = game["game"]
    half = game["half"]
    data_path = game["data_path"]
    frame_index2action = game.get("frame_index2action", {})

    pred_path = os.path.join(pred_dir, f"{half}_{game_id.replace(os.sep, '_')}_predictions.json")
    if not os.path.exists(pred_path):
        print(f"[!] Missing prediction file: {pred_path}")
        continue

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    data = np.load(data_path)
    total_frames = len(data)

    label_array = np.zeros(total_frames, dtype=int)
    for frame_idx, class_name in frame_index2action.items():
        if 0 <= frame_idx < total_frames and class_name in constants.classes:
            label_array[frame_idx] = constants.class2target[class_name]

    pred_array = np.zeros(total_frames, dtype=int)
    for frame_str, pred in predictions.items():
        frame_idx = int(frame_str)
        label = pred["label"]
        if (
            0 <= frame_idx < total_frames
            and label in constants.classes
            and pred["confidence"] > 0.7
        ):
            pred_array[frame_idx] = constants.class2target[label]

    event_indices = np.where(label_array != 0)[0]
    all_indices = set(range(total_frames))

    near_event = set()
    for idx in event_indices:
        for offset in range(-sequence, sequence + 1):
            f = idx + offset
            if 0 <= f < total_frames:
                near_event.add(f)

    background_indices = list(all_indices - near_event)

    if len(event_indices) == 0 or len(background_indices) == 0:
        continue

    sampled_background = np.random.choice(background_indices, size=len(event_indices), replace=False)

    selected_indices = np.concatenate([event_indices, sampled_background])
    selected_indices.sort()

    all_labels.extend(label_array[selected_indices].tolist())
    all_preds.extend(pred_array[selected_indices].tolist())

print("=== Evaluation Result on Balanced Test Set ===")
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(
    all_labels,
    all_preds,
    labels=range(len(constants.classes)),
    target_names=constants.classes
))

