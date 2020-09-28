
import gpflow as gp
import tensorflow as tf
import numpy as np

def gp_sample_funct(xt, cost_kernel, Xt, Yt_cost, noise_cost):

    log_Yt_cost = np.log(Yt_cost)
    latent_cost_model = gp.models.GPR((Xt, log_Yt_cost), cost_kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    yt_latent= latent_cost_model.predict_f_samples(xt)

    yt = tf.exp(yt_latent); yt= yt.numpy()

    return yt


