import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # np.zeros_like(z) initializes a copy of z that's full of zeros.
        res = np.zeros_like(z)
        for i in range(len(z)):
            # Formula: 1 / (1 + e^(-z))
            res[i] = 1 / (1 + np.exp(-z[i]))
        return np.round(res, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # np.zeros_like(z) initializes a copy of z that's full of zeros.
        res = np.zeros_like(z)
        for i in range(len(z)):
            # Formula: max(0, z) element-wise
            res[i] = max(0, z[i])
        return res