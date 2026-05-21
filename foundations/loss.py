import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        y_pred += 1e-7
        # 'mask' is our condition for np.where.
        mask = y_true == 1
        # np.where outputs an array with elements containing the log values corresponding
        #   to each sample, which are calculated differently depending on whether the true
        #       label is 1 (np.log(y_pred)) or 0 (np.log(1 - y_pred)).
        logs_of_samples = np.where(mask, np.log(y_pred), np.log(1 - y_pred))
        log_sum = np.sum(logs_of_samples)
        # return round(your_answer, 4)
        return np.round(-log_sum / np.size(y_pred), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        y_pred += 1e-7
        mask = y_true == 1
        # np.where outputs an array with elements containing either the log values corresponding
        #   to each sample where the true label is 1, or 0 otherwise.
        logs_of_samples = np.where(mask, np.log(y_pred), 0)
        log_sum = np.sum(logs_of_samples)
        # return round(your_answer, 4)
        # y_pred.shape[0] corresponds to the no. of nested arrays (samples) in y_pred.
        return np.round(-log_sum / y_pred.shape[0], 4)
