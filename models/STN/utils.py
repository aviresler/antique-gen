import numpy as np


def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def get_initial_weights_translation_only(output_size):
    b = np.zeros((4, 1), dtype='float32')
    b[0, 0] = -0.25
    b[1, 0] = -0.25
    b[2, 0] = 0.25
    b[3, 0] = 0.25
    W = np.zeros((output_size, 4), dtype='float32')
    weights = [W, b.flatten()]
    return weights
