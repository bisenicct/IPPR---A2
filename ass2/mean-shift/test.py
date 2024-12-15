import numpy as np

weights = np.array([[0.2, 0.5, 0.3], 
                    [0.1, 0.6, 0.3]])

shifted = np.array([[1, 2],
                    [3, 4],
                    [5, 6]])

# Expand dimensions of weights for broadcasting
weighted_sum = (weights[:, :, None] * shifted).sum(1)

print(weighted_sum.shape)
