import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src import constants
from configs import action

class ActionDataset(Dataset):
    def __init__(self, data, num_classes=constants.num_classes):
        self.data = data
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        half = self.data[idx]["half"]
        data_path = self.data[idx]["data_path"]
        frame_index2action = self.data[idx]["frame_index2action"]

        data_game = np.load(data_path)

        fixed_length = action.fixed_length
        if data_game.shape[0] < fixed_length:
            pad_size = fixed_length - data_game.shape[0]
            padding = np.zeros((pad_size, data_game.shape[1]))
            data_game = np.concatenate([data_game, padding], axis=0)
        elif data_game.shape[0] > fixed_length:
            data_game = data_game[:fixed_length]

        total_frames = data_game.shape[0]

        label_matrix = np.zeros((total_frames, self.num_classes))

        for frame_idx, event in frame_index2action.items():
            class_idx = constants.class2target[event]
            if frame_idx < total_frames:
                label_matrix[frame_idx, class_idx] = 1

        features = torch.tensor(data_game, dtype=torch.float32)
        labels = torch.tensor(label_matrix, dtype=torch.float32)

        return features, labels
