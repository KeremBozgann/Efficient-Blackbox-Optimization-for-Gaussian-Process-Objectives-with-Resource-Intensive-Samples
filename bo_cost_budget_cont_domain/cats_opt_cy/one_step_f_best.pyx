
import numpy as np
cimport numpy as np

def one_step_f_best(float f_best, np.ndarray u_x1, np.ndarray var_x1, np.ndarray Z):

    cdef np.ndarray L0_x1, f_x1
    cdef float f_x1min, f_best1

    L0_x1= np.linalg.cholesky(var_x1)

    f_x1= u_x1+ np.matmul(L0_x1,Z)

    f_x1min= np.min(f_x1)
    f_best1= np.minimum(f_best, f_x1min)

    return f_best1