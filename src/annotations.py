import json
from pathlib import Path

import numpy as np

from src import constants

def get_game_data(game: str, only_visible=True) -> list[dict]:

    game_dir = constants.soccernet_dir / game
    labels_json_path = game_dir / "Labels-v2.json"
    with open(labels_json_path) as file:
        labels = json.load(file)

    annotations = labels["annotations"]

    halves_set = set()
    for annotation in annotations:
        half = int(annotation["gameTime"].split(" - ")[0])
        halves_set.add(half)
        annotation["half"] = half
    halves = sorted(halves_set)

    half2video_data = dict()
    for half in halves:
        half_game_path = str(game_dir / f"{half}_ResNET_TF2_PCA512.npy")
        half2video_data[half] = dict(
            data_path=half_game_path,
            game=game,
            half=half,
            frame_index2action=dict(),
        )

    for annotation in annotations:
        if only_visible and annotation["visibility"] != "visible":
            continue
        video_data = half2video_data[annotation["half"]]
        frame_index = round(float(annotation["position"]) * constants.video_fps * 0.001)
        label = annotation["label"]
        if label in constants.card_classes:
            video_data["frame_index2action"][frame_index] = "Card"
        else:
            video_data["frame_index2action"][frame_index] = label

    return list(half2video_data.values())

def get_data(games: list[str],
                    only_visible=True) -> list[dict]:
    games_data = list()
    for game in games:
        games_data += get_game_data(
            game,
            only_visible=only_visible,
        )
    return games_data