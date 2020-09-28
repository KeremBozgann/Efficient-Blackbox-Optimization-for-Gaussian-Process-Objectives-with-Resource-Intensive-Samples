
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import gpflow as gp
from gpflow.utilities import print_summary

def ackley(x):

    firstSum = np.mean(x**2.0, axis=1).reshape(-1, 1)
    secondSum = np.mean(np.cos(2.0*np.pi*x)).reshape(-1, 1)
    result= -20.0*np.exp(-0.2*np.sqrt(firstSum)) - np.exp(secondSum) + 20 + np.e

    return np.atleast_2d(result)


def ackley_opt(D):

    y_opt= 0
    x_opt= [(0.0,) * D]
    domain= []
    for i in range(D):
        domain.append([-32.768, 32.768])

    return y_opt, x_opt, domain

def ackley_grid():

    disc=101
    domain =[]
    for i in range(5):
        domain.append([-32.768, 32.768])
    # x_arr= []
    # for i in range(5):
    #     x= np.linspace(domain[i][0], domain[i][1], disc)
    #     x_arr.append(x)
    # X= np.meshgrid(*x_arr)
    #
    # X_grid= np.empty([disc**5, 5])
    # for i in range(5):
    #     X_grid[:,i]= X[i].flatten()
    # Y = exp_cos_5d(X_grid)

    X_grid_0= (np.linspace(domain[0][0], domain[0][1], disc)).reshape(-1, 1)
    zeros= np.zeros([disc, 1])
    X_grid= np.concatenate((X_grid_0, zeros, zeros, zeros, zeros), axis=1)
    Y = ackley(X_grid)

    return X_grid_0, Y


def ackley_find_best_suited_kernel(X, Y, noise=10**(-4)):

    #calculate lengthscale of slice
    Y_latent= np.log(Y)
    kernel= gp.kernels.RBF()

    model= gp.models.GPR((X, Y_latent), kernel)
    model.likelihood.variance.assign(noise)
    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)
    ls= kernel.lengthscales.numpy()
    var= kernel.variance.numpy()
    kernel_multi= gp.kernels.RBF(lengthscales= np.ones([5])*ls, variance= var)

    return kernel_multi

# print(Ackley(np.random.rand(1,5)))