import numpy as np


def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def get_initial_weights_translation_only(output_size):
    b = np.zeros((2, 1), dtype='float32')
    b[0, 0] = 0.0
    b[1, 0] = 0.0
    W = np.zeros((output_size, 2), dtype='float32')
    weights = [W, b.flatten()]
    return weights
