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
from src.indexes import StackIndexesGenerator
from torch import nn

INDEX_SAVE_ZONE = 1

model = LSTMActionSpotting().to(device)
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.eval()

save_dir = "./data/experiments/predictions/lstm"
os.makedirs(save_dir, exist_ok=True)

test_games = constants.test_games

for game in test_games:
    game_dir = soccernet_dir / game
    game_prediction_dir = save_dir / game
    os.makedirs(game_prediction_dir, exist_ok=True)
    print("Predict game:", game)

    half2class_actions = dict()
    for half in constants.halves:
        data_path = game_dir / f"{half}_ResNET_TF2_PCA512.npy"
        raw_predictions_path = game_prediction_dir / f"{half}_raw_predictions.npz"
        print("Predict data:", data_path)
        game_data = np.load(data_path)
        frame_count = game_data.shape[0]
        indexes_generator = StackIndexesGenerator(
            action.sequence,
            action.step,
        )
        min_frame_index = indexes_generator.clip_index(0, frame_count, INDEX_SAVE_ZONE)
        max_frame_index = indexes_generator.clip_index(frame_count, frame_count, INDEX_SAVE_ZONE)

        frame_index2prediction = dict()

        frame_index2frame = dict()
        stack_indexes2features = dict()

        current_index = 0
        with tqdm() as t:
            while True:
                try:
                    if current_index < frame_count - 1:
                        frame = current_index
                        current_index += 1
                    else:
                        raise RuntimeError("End of frames")
                except BaseException as error:
                    logger.error(
                        f"Error while fetching frame {index} from '{str(self.video_path)}': {error}."
                        f"Replace by empty frame."
                    )
                    # frame = torch.zeros(self.height, self.width,
                    #                     dtype=torch.uint8,
                    #                     device=f"cuda:{self.gpu_id}")
                frame_index2frame[current_index] = frame

                predict_offset = indexes_generator.make_stack_indexes(0)[-1]
                predict_index = current_index - predict_offset
                predict_indexes = indexes_generator.make_stack_indexes(predict_index)

                for index in list(frame_index2frame.keys()):
                    if index < predict_indexes[0]:
                        del frame_index2frame[index]
                for stack_indexes in list(stack_indexes2features.keys()):
                    if any([i < predict_indexes[0] for i in stack_indexes]):
                        del stack_indexes2features[stack_indexes]

                if set(predict_indexes) <= set(frame_index2frame.keys()):
                    features = []
                    with torch.no_grad():
                        for i in predict_indexes:
                            features.append(game_data[i])
                        prediction = torch.from_numpy(features).unsqueeze(0).float().to(device)
                        logits = model(prediction)
                        prediction_transform = nn.Sigmoid()
                        probs = prediction_transform(logits)
                else:
                    prediction = None

                if predict_index < min_frame_index:
                    continue
                if prediction is not None:
                    frame_index2prediction[predict_index] = prediction.cpu().numpy()
                t.update()
                if predict_index == max_frame_index:
                    break
        frame_index2frame = dict()
        stack_indexes2features = dict()

        frame_indexes = sorted(frame_index2prediction.keys())
        raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
        np.savez(
            raw_predictions_path,
            frame_indexes=frame_indexes,
            raw_predictions= raw_predictions,
        )
        print("Raw predictions saved to", raw_predictions_path)
        # class2actions = raw_predictions_to_actions(frame_indexes, raw_predictions)

        half2class_actions[half] = class_actions

    # prepare_game_spotting_results(half2class_actions, game, prediction_dir)
