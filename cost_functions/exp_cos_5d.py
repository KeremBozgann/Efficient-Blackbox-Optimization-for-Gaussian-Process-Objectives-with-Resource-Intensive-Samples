import numpy as np
import matplotlib.pyplot as plt
import gpflow as gp
from gpflow.utilities import print_summary
from mpl_toolkits.mplot3d import Axes3D

def exp_cos_5d(X):

    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)
    x3 = X[:, 2].reshape(-1, 1)
    x4 = X[:, 3].reshape(-1, 1)
    x5 = X[:, 4].reshape(-1, 1)


    Y= np.exp(np.cos(np.pi*((x1+32.768)/65.536+1))+np.cos(np.pi*((x2+32.768)/65.536+1))+np.cos(np.pi*((x3+32.768)/65.536+1))+\
              np.cos(np.pi*((x4+32.768)/65.536+1))+np.cos(np.pi*((x5+32.768)/65.536+1)))

    return Y


def exp_cos_5d_opt():

    domain =[]
    for i in range(5):
        domain.append([-32.768, 32.768])

    y_opt= np.exp(-5)
    x_opt= [-32.768, -32.768, -32.768, -32.768 ,-32.768]

    return y_opt, x_opt, domain

def exp_cos_5d_grid():

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
    Y = exp_cos_5d(X_grid)

    return X_grid_0, Y

def exp_cos_5d_find_best_suited_kernel(X, Y, noise=10**(-4)):

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