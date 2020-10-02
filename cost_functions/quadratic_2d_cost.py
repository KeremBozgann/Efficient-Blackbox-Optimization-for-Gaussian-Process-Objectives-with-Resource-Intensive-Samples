import numpy as np
import gpflow as gp
from gpflow.utilities import print_summary
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def quadratic_2d_cost(X):
    x1= X[:,0].reshape(-1,1)
    x2= X[:, 1].reshape(-1,1)

    return np.exp(2- (x1**2+x2**2))


# def quadratic_2d_cost_opt():
#
#     domain = [[-1,1], [-1,1]]
#
#     y_opt= 0
#     x_opt= [0.0, 0.0]
#
#
#     return y_opt, x_opt, domain

def quadratic_2d_cost_plots(disc, plot= False):

    domain = [[-1,1], [-1,1]]


    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    Y= quadratic_2d_cost(X)

    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('quadratic_2d_cost')
        ax.scatter3D(X[:,0], X[:,1], Y[:,0], color= 'blue', label= 'posterior target')
        plt.show()

    return X, Y

def quadratic_2d_cost_find_best_suited_kernel(X, Y, noise=10**(-4)):

    kernel= gp.kernels.RBF()
    Y_latent= np.log(Y)

    model= gp.models.GPR((X, Y_latent), kernel )
    model.likelihood.variance.assign(noise)
    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)

    return model, kernel