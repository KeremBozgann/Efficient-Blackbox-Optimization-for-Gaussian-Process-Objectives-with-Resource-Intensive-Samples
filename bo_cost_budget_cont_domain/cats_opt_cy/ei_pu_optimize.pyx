
import numpy as np
from one_step_f_best import one_step_f_best


def EI_x2_pu_negative(np.ndarray x2, ):

    cdef np.ndarray u0_x2, var0_x2
    # print('x2 shape before reshaping in obj', x2.shape)
    x2= x2.reshape(1,-1)
    # print('x2 shape in obj', x2.shape)
    u0_x2_tens, var0_x2_tens= model.predict_f(x2); u0_x2= u0_x2_tens.numpy(); var0_x2= var0_x2_tens.numpy()

    u0_x2_latent_cost, var0_x2_latent_cost= latent_cost_model.predict_f(x2); u0_x2_latent_cost= u0_x2_latent_cost.numpy();
    var0_x2_latent_cost= var0_x2_latent_cost.numpy()
    u0_x2_cost= np.exp(u0_x2_latent_cost)

    u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)

    sigma1_x2= np.sqrt(var1_x2)

    EI1_x2_pu= EI_pu(sigma1_x2, u1_x2 , f_best1, u0_x2_cost)

    # print('EI1 shape before reshaping in obj', EI1_x2.shape)
    EI1_x2_pu= EI1_x2_pu.flatten()
    # print('ei1 shape in obj',EI1_x2.shape)
    return -EI1_x2_pu

def EI1_x2_per_cost_optimize(np.ndarray u0_x1, np.ndarray var0_x1, kernel, np.ndarray x1, np.ndarray Z,
                                         np.ndarray Xt, np.ndarray Yt, float noise, model, latent_cost_model, float f_best):


    '''observation at x1'''

    cdef np.ndarray sigma0_x1, L0_x1, y1, Xt1, Yt1
    float f_best1

    sigma0_x1= np.sqrt(var0_x1)
    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z)
    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    f_best1 = one_step_f_best(f_best, u0_x1, var0_x1, Z_conv)
