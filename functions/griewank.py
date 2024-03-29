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
from util import create_grid

import sys
from scipy.optimize import Bounds
from scipy.optimize import minimize

sys.path.append('../bo_cost_budget_cont_domain')
from hyperparameter_optimization import logistic_bjt

def griewank(X):

    # x1= X[:,0].reshape(-1,1); x2= X[:,1].reshape(-1,1)
    term1 = 0
    term2 = 1

    for i in range(X.shape[1]):
        xi = X[:, i].reshape(-1, 1)
        term1 += xi ** 2
        term2 *= np.cos(xi / np.sqrt(i + 1))
    result = term1 / 4000 - term2 + 1

    return result

def griewank_opt(D):

    y_opt= 0.0
    x_opt= [[0.0]]*D
    domain=  [[-5,5]]*D

    return y_opt, x_opt, domain


def scipy_minimize_griewank(D):

    def fun(x):
        x= x.reshape(1,-1)

        term1 = 0
        term2 = 1

        for i in range(x.shape[1]):
            xi = x[:, i].reshape(-1, 1)
            term1 += xi ** 2
            term2 *= np.cos(xi / np.sqrt(i + 1))
        result = term1 / 4000 - term2 + 1

        result= result.flatten()

        return result

    domain=  [[-5,5]]*D

    lower = [];upper = []

    for i in range(len(domain)):
        lower.append(domain[i][0]); upper.append(domain[i][1])
    b = Bounds(lb=lower, ub=upper)

    x_opt_list= np.empty([100, D])
    value_list= np.empty([100, 1])

    for i in range(100):

        x0 = np.random.uniform(lower, upper, (1, D))
        result= minimize(fun= fun, bounds= b, x0= x0, method= 'L-BFGS-B')
        # print('optimum point:{}, optimum value:{}'.format(result['x'], result['fun']))
        x_opt_list[i, :] = result['x'].reshape(1,-1)
        value_list[i, :]= result['fun'].reshape(1,-1)
    index= np.argmin(value_list, axis=0)
    print('optimum value:{}, optimum point:{}'.format(value_list[index, :].reshape(-1,1), x_opt_list[index , :].reshape(1,-1)))


def griewank_plots(disc, D):

    domain=  [[-5,5]]*D
    X= create_grid(disc, domain)

    Y= griewank(X)

    return X, Y

def griewank_find_best_suited_kernel(X, Y, noise=10**(-4)):

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