import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src import constants
from configs import action
import random
from collections import defaultdict
from src.indexes import StackIndexesGenerator, FrameIndexShaker
from typing import Callable, Type, Optional
from src.utils import set_random_seed
from src.target import VideoTarget

class TrainActionDataset(Dataset):
    def __init__(
            self,
            videos_data: list[dict],
            classes: list[str],
            indexes_generator: StackIndexesGenerator,
            videos_sampling_weights: list[np.ndarray],
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frame_index_shaker: Optional[FrameIndexShaker] = None,

    ):
        self.videos_data = videos_data
        self.data_games = [
            np.load(v["data_path"]) for v in self.videos_data
        ]
        self.num_videos = len(self.videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_sampling_weights = videos_sampling_weights
        self.videos_frame_indexes = [np.arange(v["frame_count"]) for v in videos_data]
        self.frame_index_shaker = frame_index_shaker
        self.indexes_generator=indexes_generator
        self.videos_target = [
            VideoTarget(data, classes) for data in self.videos_data
        ]
        self.target_process_fn = target_process_fn

    def __len__(self) -> int:
        return self.num_actions

    def get_video_frame_indexes(self, index) -> tuple[int, list[int]]:
        set_random_seed(index)
        video_index = random.randrange(0, self.num_videos)
        frame_index = np.random.choice(self.videos_frame_indexes[video_index],
                                       p=self.videos_sampling_weights[video_index])
        save_zone = 0
        if self.frame_index_shaker is not None:
            save_zone += max(abs(sh) for sh in self.frame_index_shaker.shifts)
        frame_index = self.indexes_generator.clip_index(
            frame_index, self.videos_data[video_index]["frame_count"], save_zone
        )
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        if self.frame_index_shaker is not None:
            frame_indexes = self.frame_index_shaker(frame_indexes)
        return video_index, frame_indexes

    def get_targets(self, video_index: int, frame_indexes: list[int]):
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        targets = self.videos_target[video_index].targets(target_indexes)
        return targets

    def __getitem__(self, idx):
        video_index, frame_indexes = self.get_video_frame_indexes(idx)

        video_data = self.videos_data[video_index]
        data_game = self.data_games[video_index]
        frames = []
        for i in frame_indexes:
            frames.append(data_game[i])

        targets = self.get_targets(video_index, frame_indexes)
        targets = targets.astype(np.float32, copy=False)
        num_crop_targets = targets.shape[0] - action.sequence
        left = num_crop_targets // 2
        right = num_crop_targets - left
        target =  targets[left:-right]
        target = np.amax(targets, axis=0)

        return torch.tensor(frames, dtype=torch.float32), torch.from_numpy(target)

class ValActionDataset(Dataset):
    def __init__(
            self,
            videos_data: list[dict],
            classes: list[str],
            indexes_generator: StackIndexesGenerator,
            videos_sampling_weights: list[np.ndarray],
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frame_index_shaker: Optional[FrameIndexShaker] = None,

    ):
        self.videos_data = videos_data
        self.data_games = [
            np.load(v["data_path"]) for v in self.videos_data
        ]
        self.num_videos = len(self.videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_sampling_weights = videos_sampling_weights
        self.videos_frame_indexes = [np.arange(v["frame_count"]) for v in videos_data]
        self.frame_index_shaker = frame_index_shaker
        self.indexes_generator=indexes_generator
        self.videos_target = [
            VideoTarget(data, classes) for data in self.videos_data
        ]
        self.target_process_fn = target_process_fn

    def __len__(self) -> int:
        return self.num_actions

    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        assert 0 <= index < self.__len__()
        action_index = index
        video_index = 0
        for video_index, num_video_actions in enumerate(self.num_videos_actions):
            if action_index >= num_video_actions:
                action_index -= num_video_actions
            else:
                break
        video_target = self.videos_target[video_index]
        video_data = self.videos_data[video_index]
        frame_index = video_target.get_frame_index_by_action_index(action_index)
        frame_index = self.indexes_generator.clip_index(frame_index, video_data["frame_count"], 1)
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return video_index, frame_indexes

    def get_targets(self, video_index: int, frame_indexes: list[int]):
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        targets = self.videos_target[video_index].targets(target_indexes)
        return targets

    def __getitem__(self, idx):
        video_index, frame_indexes = self.get_video_frame_indexes(idx)

        video_data = self.videos_data[video_index]
        data_game = self.data_games[video_index]
        frames = []
        for i in frame_indexes:
            frames.append(data_game[i])

        targets = self.get_targets(video_index, frame_indexes)
        targets = targets.astype(np.float32, copy=False)
        num_crop_targets = targets.shape[0] - action.sequence
        left = num_crop_targets // 2
        right = num_crop_targets - left
        target =  targets[left:-right]
        target = np.amax(targets, axis=0)

        return torch.tensor(frames, dtype=torch.float32), torch.from_numpy(target)
