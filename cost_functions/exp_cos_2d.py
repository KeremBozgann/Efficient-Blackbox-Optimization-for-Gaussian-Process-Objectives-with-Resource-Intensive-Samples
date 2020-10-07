import numpy as np
import matplotlib.pyplot as plt
import gpflow as gp
from gpflow.utilities import print_summary
from mpl_toolkits.mplot3d import Axes3D

def exp_cos_2d(X):

    x1 = X[:, 0].reshape(-1, 1);
    x2 = X[:, 1].reshape(-1, 1)

    Y= np.exp(np.cos(np.pi*x1)*np.cos(np.pi*x2))


    return Y


def exp_cos_2d_opt():

    domain= [[-5,10], [0,15]]

    y_opt= -1
    x_opt= [-0.5, 1.5]


    return y_opt, x_opt, domain

def exp_cos_2d_plots(disc, plot=False):

    domain= [[-5,10], [0,15]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    # x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    Y = exp_cos_2d(X)

    if plot == True:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('cos2d')

        ax.scatter3D(X[:, 0], X[:, 1], Y[:, 0], color='blue', label='posterior target')
        plt.show()

    return X, Y

def exp_cos_2d_find_best_suited_kernel(X, Y, noise=10**(-4)):

    Y_latent= np.log(Y)
    kernel= gp.kernels.RBF()

    model= gp.models.GPR((X, Y_latent), kernel )
    model.likelihood.variance.assign(noise)
    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)

    return model, kernel