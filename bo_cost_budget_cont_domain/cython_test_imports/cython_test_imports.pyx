import numpy as np
cimport numpy as np
import gpflow



def CATS_negative_improvement(np.ndarray x1, np.ndarray Z,  model,  latent_cost_model,
                                np.ndarray Xt, np.ndarray Yt, float f_best,  kernel, float noise):

    cdef np.ndarray u0_x1, var0_x1, sigma0_x1, u0_x1_cost_latent, var0_x1_cost_latent, sigma0_x1_cost_latent, L0_x1, \
                        y1, Xt1, Yt1

    float i0_x1

    x1 = x1.reshape(1, -1)
    u0_x1_tens, var0_x1_tens = model.predict_f(x1);
    u0_x1 = u0_x1_tens.numpy();
    var0_x1 = var0_x1_tens.numpy()
    sigma0_x1 = np.sqrt(var0_x1)

    u0_x1_cost_latent_tens, var0_x1_cost_latent_tens = latent_cost_model.predict_f(x1);
    u0_x1_cost_latent = u0_x1_cost_latent_tens.numpy();
    var0_x1_cost_latent = var0_x1_cost_latent_tens.numpy()
    sigma0_x1_cost_latent = np.sqrt(var0_x1_cost_latent)

    L0_x1 = np.linalg.cholesky(var0_x1)
    y1 = u0_x1 + np.matmul(L0_x1, Z)
    '''update Xt and Yt according to outcome of Z'''
    Xt1 = np.append(Xt, x1, axis=0)
    Yt1 = np.append(Yt, y1, axis=0)

    i0_x1 = np.maximum(f_best - float(np.min(Yt1, axis=0)), 0)

    x2_opt_value, x2_opt, ei1_x2_opt = EI1_x2_per_cost_optimize(u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model,
                                                    latent_cost_model, f_best)

    return u0_x1, var0_x1, sigma0_x1, i0_x1