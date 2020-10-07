import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gpflow as gp
from gpflow.utilities import print_summary


def logistic_bjt(lower, upper):
    '''define logistic bijector'''
    low= tf.constant(lower, dtype= tf.float64); high= tf.constant(upper, dtype= tf.float64)
    affine = tfp.bijectors.AffineScalar(shift=low.numpy(), scale=(high.numpy() - low.numpy()))
    sigmoid = tfp.bijectors.Sigmoid()
    logistic = tfp.bijectors.Chain([affine, sigmoid])
    return logistic

def set_and_optimize_gp_model(optimize, D, Xt, Yt, Yt_cost, noise, noise_cost, kernel, latent_cost_kernel, logistic, logistic_noise):

    if optimize:

        kernel= gp.kernels.RBF(lengthscales= np.array([1]*D))
        latent_cost_kernel= gp.kernels.RBF(lengthscales= np.array([1]*D))

        model = gp.models.GPR((Xt, Yt), kernel=kernel)
        '''set hyperparameter constraints'''
        model.kernel.lengthscales = gp.Parameter(model.kernel.lengthscales.numpy(), transform=logistic)
        model.kernel.variance = gp.Parameter(model.kernel.variance.numpy(), transform=logistic)
        model.likelihood.variance = gp.Parameter(model.likelihood.variance.numpy(), transform=logistic_noise)

        opt_obj = gp.optimizers.Scipy()
        opt_obj.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))


        log_Yt_cost = np.log(Yt_cost)
        latent_cost_model = gp.models.GPR((Xt, log_Yt_cost), latent_cost_kernel)
        '''set hyperparameter constraints'''
        latent_cost_model.kernel.lengthscales = gp.Parameter(latent_cost_model.kernel.lengthscales.numpy(), transform=logistic)
        latent_cost_model.kernel.variance = gp.Parameter(latent_cost_model.kernel.variance.numpy(), transform=logistic)
        latent_cost_model.likelihood.variance = gp.Parameter(latent_cost_model.likelihood.variance.numpy(), transform=logistic_noise)

        opt_cost = gp.optimizers.Scipy()
        opt_cost.minimize(latent_cost_model.training_loss, latent_cost_model.trainable_variables, options=dict(maxiter=100))

        noise= model.likelihood.variance.numpy()
        noise_cost= latent_cost_model.likelihood.variance.numpy()

        print('printing objective model summary')
        print_summary(model)
        print('lengthscale values of objective model: ',model.kernel.lengthscales)
        print('printing latent cost model summary')
        print_summary(latent_cost_model)
        print('lengthscale values of latent cost model: ',model.kernel.lengthscales)


    else:

        model = gp.models.GPR((Xt, Yt), kernel=kernel)
        model.likelihood.variance.assign(noise)

        log_Yt_cost = np.log(Yt_cost)
        latent_cost_model = gp.models.GPR((Xt, log_Yt_cost), latent_cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

    return model, latent_cost_model, noise, noise_cost, log_Yt_cost, kernel, latent_cost_kernel
