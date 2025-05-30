import SoccerNet
from SoccerNet.utils import getListGames

from pathlib import Path

work_dir = Path(".")
soccernet_dir = work_dir / "data" / "soccernet" / "ResNET"

train_games = getListGames(split="train", task="spotting", dataset="SoccerNet")
valid_games = getListGames(split="valid", task="spotting", dataset="SoccerNet")
test_games = getListGames(split="test", task="spotting", dataset="SoccerNet")

split2games = {
    "train": train_games,
    "valid": valid_games,
    "test": test_games,
}

classes = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Card",
]
card_classes = [
    "Yellow card",
    "Red card",
    "Yellow->red card",
]

num_classes = len(classes)
target2class: dict[int, str] = {trg: cls for trg, cls in enumerate(classes)}
class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}

video_fps = 2
