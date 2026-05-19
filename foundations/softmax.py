import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Subtracted max(z) for numerical stability before computing exp
        z = z - max(z)
        # np.exp() turns z = [i, j, k] into z = [e^i, e^j, e^k]
        z = np.exp(z)
        # np.sum() will sum up all elements of z
        sum = np.sum(z)
        # Divides each  by the sum, then rounds to 4 decimal places
        return np.round(z / sum, 4)
