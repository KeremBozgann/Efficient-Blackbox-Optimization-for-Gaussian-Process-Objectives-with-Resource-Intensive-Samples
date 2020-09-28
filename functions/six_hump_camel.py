
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import gpflow as gp
from gpflow.utilities import print_summary

def camel(X):

    X1= X[:,0].reshape(-1,1); X2= X[:,1].reshape(-1,1)

    result = (4-2.1*X1**2+ X1**4/3)*X1**2+ X1*X2+ (-4+4*X2**2)*X2**2

    return result

def camel_opt():

    domain = [[-3,3],[-2,2]]

    y_opt= -1.0316
    x_opt= [[0.0898, -0.7126],[-0.0898, 0.7126]]


    return y_opt, x_opt, domain

def camel_plots(disc, plot=False):



    domain = [[-3,3],[-2,2]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1.reshape(-1, 1), X2.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    Y= camel(X)

    if plot== True:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('camel')
        ax.scatter3D(X[:,0], X[:,1], Y[:,0], color= 'blue', label= 'true objective')
        plt.show()

    return X, Y

def camel_find_best_suited_kernel(X, Y, noise=10**(-4)):

    kernel= gp.kernels.Matern52()

    model= gp.models.GPR((X, Y), kernel )
    model.likelihood.variance.assign(noise)
    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)

    return model, kernel