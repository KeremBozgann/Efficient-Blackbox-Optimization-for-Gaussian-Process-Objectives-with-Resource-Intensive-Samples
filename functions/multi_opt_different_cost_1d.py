
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import gpflow as gp
from gpflow.utilities import print_summary

import sys
sys.path.append('../bo_cost_budget_cont_domain')
from hyperparameter_optimization import logistic_bjt

def multi_opt_1d(X):


    result = np.sin(np.pi*X)+ np.sin(1.5*np.pi*X) +np.sin(2*np.pi*X)

    # result = float(result)

    # print('Result = %f' % result)
    # time.sleep(np.random.randint(60))
    return result

def multi_opt_1d_opt():

    domain = [[-3,3]]

    y_opt= -2.74776185
    x_opt= [-0.3]


    return y_opt, x_opt, domain

def multi_opt_1d_plots(disc, plot= False):



    domain =[[-3,3]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    X= x1.reshape(-1,1)

    Y= multi_opt_1d(X)

    if plot== True:
        plt.figure()
        plt.plot(X[:,0], Y[:,0], color= 'blue', label= 'true target')
        plt.show()
    return X, Y

def multi_opt_1d_find_best_suited_kernel(X, Y, noise=10**(-4)):

    '''constraint values'''
    lower= 10**(-5); upper= 10**(6); #lengtscale and variance constarint
    lower_noise= 10**(-5); upper_noise= 10**(6); #noise constarint

    logistic = logistic_bjt(lower, upper)
    logistic_noise = logistic_bjt(lower_noise, upper_noise)

    D= X.shape[1]
    kernel = gp.kernels.RBF(lengthscales=np.array([1] * D))

    model = gp.models.GPR((X, Y), kernel=kernel)
    '''set hyperparameter constraints'''
    model.kernel.lengthscales = gp.Parameter(model.kernel.lengthscales.numpy(), transform=logistic)
    model.kernel.variance = gp.Parameter(model.kernel.variance.numpy(), transform=logistic)
    # model.likelihood.variance = gp.Parameter(model.likelihood.variance.numpy(), transform=logistic_noise)

    model.likelihood.variance.assign(noise)

    gp.set_trainable(model.likelihood, False)

    opt = gp.optimizers.Scipy()

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    print_summary(model)

    return model, kernel