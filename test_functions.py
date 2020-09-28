import numpy as np

def branin(x1, x2):

    y = np.square(x2 - (5.1 / (4 * np.square(np.pi))) * np.square(x1) +
                       (5 / np.pi) * x1 - 6) + 10 * (1 - (1. / (8 * np.pi))) * np.cos(x1) + 10

    y = float(y)

    return y

def six_hump_camel(x1, x2):

    y= (4-2.1*x1**2+ x1**4/3)*x1**2+x1*x2+(-4+4*x2**2)*x2**2

    return y


