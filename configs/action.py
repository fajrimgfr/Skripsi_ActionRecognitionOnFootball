batch_size = 4
num_epochs = 20
learning_rate = 3e-4
device = "cuda"
stages = "train"
sequence = 5
step = 1

shifts =[-1, 0, 1]
weights = [0.2, 0.6, 0.2]
prob = 0.25


action_window_size=5
action_prob=0.5
action_weights={
    "Penalty": 0.244,
    "Kick-off": 0.197,
    "Goal": 0.08,
    "Substitution": 0.06,
    "Offside": 0.069,
    "Shots on target": 0.028,
    "Shots off target": 0.03,
    "Clearance": 0.041,
    "Ball out of play": 0.011,
    "Throw-in": 0.015,
    "Foul": 0.017,
    "Indirect free-kick": 0.028,
    "Direct free-kick": 0.077,
    "Corner": 0.035,
    "Card": 0.07,
}