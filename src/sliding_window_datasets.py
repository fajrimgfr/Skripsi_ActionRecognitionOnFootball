import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src import constants
from configs import action
import random

class SlidingWindowDataset(Dataset):
    def __init__(self, data, sequence=action.sequence, num_classes=constants.num_classes):
        self.samples = []
        self.sequence = sequence
        self.num_classes = num_classes

        for item in data:
            data_path = item["data_path"]
            frame_index2action = item["frame_index2action"]
            data_game = np.load(data_path)
            total_frames = data_game.shape[0]

            event = set()
            for frame, label_str in frame_index2action.items():
                if frame < total_frames:
                    label = constants.class2target.get(label_str, 0)
                    self.samples.append((data_path, frame, label))
                    event.add(frame)
            
            all_frames = set(range(total_frames))
            background_frames = list(all_frames - event)

            num_event = len(frame_index2action)
            num_background = len(background_frames)

            background_frames_sampled = random.sample(background_frames, min(num_event, num_background))

            for f in sorted(background_frames_sampled):
                self.samples.append((data_path, f, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path, frame, label = self.samples[idx]
        data_game = np.load(data_path)
        feat = data_game[frame].astype(np.float32)
        return torch.tensor(feat), torch.tensor(label)
