import numpy as np

"""
    # sigmoid(40) = 1.0, 当x>=40时，将导致sigmoid恒为1
"""
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

print()

