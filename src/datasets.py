import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src import constants
from configs import action
import random

class ActionDataset(Dataset):
    def __init__(self, data, sequence=action.sequence, num_classes=constants.num_classes):
        self.samples = []
        self.num_classes = num_classes
        self.sequence = sequence

        for item in data:
            data_path = item["data_path"]
            frame_index2action = item["frame_index2action"]
            data_game = np.load(data_path)
            total_frames = data_game.shape[0]

            # Convert action label to integer index (optional: cache beforehand)
            frame_index2target = {
                int(frame): constants.class2target.get(label, 0)
                for frame, label in frame_index2action.items()
            }

            # Sliding window
            stride = int(0.5 * sequence)
            for start in range(0, total_frames - sequence + 1, stride):
                end = start + sequence

                labelList = [0] * num_classes
                for frame in range(start, end):
                    if frame in frame_index2target:
                        label_idx = frame_index2target[frame]
                        labelList[label_idx] = 1

                self.samples.append((data_path, start, labelList))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path, start, label = self.samples[idx]
        data_game = np.load(data_path)
        window = data_game[start:start + self.sequence]

        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
