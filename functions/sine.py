
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import gpflow as gp
from gpflow.utilities import print_summary

def sin(X):


    result = np.sin(np.pi*X)

    # result = float(result)

    # print('Result = %f' % result)
    # time.sleep(np.random.randint(60))
    return result

def sin_opt():

    domain = [[-2,2]]

    y_opt= -1
    x_opt= [-0.5, 1.5]


    return y_opt, x_opt, domain

def sin_plots(disc, plot= False):



    domain =[[-2,2]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)

    x1_max, x1_min = np.max(x1), np.min(x1)
    X= x1.reshape(-1,1)

    Y= sin(X)
    if plot== True:
        plt.figure()
        plt.plot(X[:,0], Y[:,0], color= 'blue', label= 'true target')
        plt.show()
    return X, Y

def sin_find_best_suited_kernel(X, Y, noise=10**(-4)):

    kernel= gp.kernels.RBF()

    model= gp.models.GPR((X, Y), kernel )
    model.likelihood.variance.assign(noise)
    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)

    return model, kernel