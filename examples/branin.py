
import numpy as np
import gpflow as gp
from scipy.stats import norm
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
import math

def branin(x1, x2):

    y = np.square(x2 - (5.1 / (4 * np.square(np.pi))) * np.square(x1) +
                       (5 / np.pi) * x1 - 6) + 10 * (1 - (1. / (8 * np.pi))) * np.cos(x1) + 10

    y = float(y)

    return y



