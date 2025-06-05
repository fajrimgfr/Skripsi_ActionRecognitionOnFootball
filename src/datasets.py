import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src import constants
from configs import action

class ActionDataset(Dataset):
    def __init__(self, data, context=9, num_classes=constants.num_classes):
        self.samples = []
        self.num_classes = num_classes
        self.context = context

        for item in data:
            data_path = item["data_path"]
            frame_index2action = item["frame_index2action"]
            total_frames = np.load(data_path).shape[0]

            event_frames = set(map(int, frame_index2action.keys()))
            all_frames = set(range(total_frames))

            near_event = set()
            for frame_idx in event_frames:
                for offset in range(-context, context + 1):
                    f = frame_idx + offset
                    if 0 <= f < total_frames:
                        near_event.add(f)

            background_frames = list(all_frames - near_event)

            for f in sorted(near_event):
                label = constants.class2target.get(frame_index2action.get(str(f), "No-event"), 0)
                self.samples.append((data_path, f, label))

            for f in sorted(background_frames):
                self.samples.append((data_path, f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path, center_idx, label = self.samples[idx]
        data_game = np.load(data_path)

        start = center_idx - self.context
        end = center_idx + self.context + 1

        feat_dim = data_game.shape[1]
        window = np.zeros((2 * self.context + 1, feat_dim), dtype=np.float32)

        for i, f in enumerate(range(start, end)):
            if 0 <= f < data_game.shape[0]:
                window[i] = data_game[f]

        return torch.tensor(window), torch.tensor(label)
