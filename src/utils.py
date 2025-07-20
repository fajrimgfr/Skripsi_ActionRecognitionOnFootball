import time
import random
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

def set_random_seed(index: int):
    seed = int(time.time() * 1000.0) + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))

def post_processing(frame_indexes: list[int],
                    predictions: np.ndarray,
                    gauss_sigma: float,
                    height: float,
                    distance: int) -> tuple[list[int], list[float]]:
    predictions = gaussian_filter(predictions, gauss_sigma)
    peaks, _ = find_peaks(predictions, height=height, distance=distance)
    confidences = predictions[peaks].tolist()
    action_frame_indexes = (peaks + frame_indexes[0]).tolist()
    return action_frame_indexes, confidences