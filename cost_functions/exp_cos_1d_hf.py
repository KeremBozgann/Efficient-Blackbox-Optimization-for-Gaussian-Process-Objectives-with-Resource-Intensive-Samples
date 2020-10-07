import numpy as np
import matplotlib.pyplot as plt
import gpflow as gp
from gpflow.utilities import print_summary

def exp_cos_1d_hf(X, plot=False):

    Y= np.exp(np.cos(2*np.pi*X))

    return Y


def exp_cos_1d_hf_opt():

    domain = [[-2,2]]

    y_opt= 1/np.e
    x_opt= [-1.0, 1.0]


    return y_opt, x_opt, domain

def exp_cos_1d_hf_plots(disc, plot=False):

    domain = [[-2, 2]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x1_max, x1_min = np.max(x1), np.min(x1)
    X = x1.reshape(-1, 1)

    Y = exp_cos_1d_hf(X)

    if plot == True:
        plt.figure()
        plt.title('cos_1d')
        plt.plot(X[:, 0], Y[:, 0], color='blue', label='true target')
        plt.show()

    return X, Y

def exp_cos_1d_hf_find_best_suited_kernel(X, Y, noise=10**(-4)):

    Y_latent= np.log(Y)
    kernel= gp.kernels.RBF()

    model= gp.models.GPR((X, Y_latent), kernel )
    model.likelihood.variance.assign(noise)
    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)

    return model, kernel