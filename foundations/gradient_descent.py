class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        if iterations == 0:
            return init
        if learning_rate < 0 or learning_rate > 1:
            raise ValueError("Gradient Descent needs a learning rate that's strictly between 0 and 1.")
        float_init = float(init)
        # Objective function: f(x) = x^2
        for x in range(iterations):
            # Derivative:         f'(x) = 2x
            deriv = 2 * float_init
            # Update rule:        x = x - learning_rate * f'(x)
            new_init = float_init - learning_rate * deriv
            float_init = new_init
        # Round final answer to 5 decimal places
        final_value = round(float_init, 5)
        return final_value