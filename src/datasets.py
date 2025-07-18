import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src import constants
from configs import action
import random
from collections import defaultdict

class ActionDataset(Dataset):
    def __init__(self, data, sequence=action.sequence, num_classes=constants.num_classes):
        self.samples = []
        self.num_classes = num_classes
        self.sequence = sequence

        self.data_cache = {}

        for item in data:
            data_path = item["data_path"]
            frame_index2action = item["frame_index2action"]

            if data_path not in self.data_cache:
                self.data_cache[data_path] = np.load(data_path)

            data_game = self.data_cache[data_path]
            total_frames = data_game.shape[0]

            frame_index2target = {
                int(frame): constants.class2target.get(label, 0)
                for frame, label in frame_index2action.items()
            }

            stride = int(0.5 * sequence)
            for start in range(0, total_frames - sequence + 1, stride):
                labelList = [0] * num_classes
                for frame in range(start, start + sequence):
                    if frame in frame_index2target:
                        label_idx = frame_index2target[frame]
                        labelList[label_idx] = 1

                self.samples.append((data_path, start, labelList))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path, start, label = self.samples[idx]
        data_game = self.data_cache[data_path]
        window = data_game[start:start + self.sequence]

        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class ActionDatasetSMOTE(Dataset):
    def __init__(self, data, sequence=action.sequence, num_classes=constants.num_classes, oversample_factor=2):
        self.sequence = sequence
        self.num_classes = num_classes
        self.samples = []
        self.data_cache = {}
        self.samples_by_class = defaultdict(list)

        # Load and cache original data
        for item in data:
            data_path = item["data_path"]
            frame_index2action = item["frame_index2action"]

            if data_path not in self.data_cache:
                self.data_cache[data_path] = np.load(data_path)

            data_game = self.data_cache[data_path]
            total_frames = data_game.shape[0]

            frame_index2target = {
                int(frame): constants.class2target.get(label, 0)
                for frame, label in frame_index2action.items()
            }

            stride = int(0.5 * sequence)
            for start in range(0, total_frames - sequence + 1, stride):
                labelList = [0] * num_classes
                for frame in range(start, start + sequence):
                    if frame in frame_index2target:
                        label_idx = frame_index2target[frame]
                        labelList[label_idx] = 1

                sample = (data_path, start, labelList)
                self.samples.append(sample)

                # Bagi ke per kelas
                for idx, val in enumerate(labelList):
                    if val == 1:
                        self.samples_by_class[idx].append(sample)

        # --- SMOTE-like augmentation ---
        synthetic_samples = []
        for class_idx, class_samples in self.samples_by_class.items():
            if len(class_samples) < 100:  # Only oversample minority classes
                for _ in range(oversample_factor * len(class_samples)):
                    s1, s2 = random.sample(class_samples, 2)
                    x1 = self.data_cache[s1[0]][s1[1]:s1[1] + sequence]
                    x2 = self.data_cache[s2[0]][s2[1]:s2[1] + sequence]

                    alpha = random.uniform(0.2, 0.8)
                    synthetic_window = x1 + alpha * (x2 - x1)
                    label = [0] * num_classes
                    label[class_idx] = 1  # single-label untuk synthetic

                    synthetic_samples.append((synthetic_window, label))

        # Simpan semua sample
        self.synthetic_samples = synthetic_samples

    def __len__(self):
        return len(self.samples) + len(self.synthetic_samples)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            data_path, start, label = self.samples[idx]
            data_game = self.data_cache[data_path]
            window = data_game[start:start + self.sequence]
        else:
            window, label = self.synthetic_samples[idx - len(self.samples)]

        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

