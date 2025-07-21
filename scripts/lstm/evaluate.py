import json
import argparse

import numpy as np

from src.evaluate import evaluate
from src import constants

predictions_path = constants.prediction_dir
print("Predictions path", predictions_path)
games = constants.test_games
print("Evaluate games", games)

results = evaluate(
    SoccerNet_path=constants.soccernet_dir,
    Predictions_path=str(predictions_path),
    list_games=games,
    prediction_file="results_spotting.json",
    version=2,
    metric="loose",
    num_classes=17,
    label_files='Labels-v2.json',
    dataset="SoccerNet",
    framerate=2,
)

print("Average mAP (tight): ", results["a_mAP"])
print("Average mAP (tight) per class: ", results["a_mAP_per_class"])

evaluate_results_path = predictions_path / "evaluate_results.json"
results = {key: (float(value) if np.isscalar(value) else list(value))
            for key, value in results.items()}
with open(evaluate_results_path, "w") as outfile:
    json.dump(results, outfile, indent=4)
print("Evaluate results saved to", evaluate_results_path)
print("Results:", results)