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

import sys
from scipy.optimize import Bounds
from scipy.optimize import minimize

sys.path.append('../bo_cost_budget_cont_domain')
from hyperparameter_optimization import logistic_bjt

from util import get_grid

def ackley_cost(X):

    D= X.shape[1]
    term1 = 0
    term2 = 0
    a= 20
    b= 0.2
    c= np.pi*2

    for i in range(D):
        xi = X[:, i].reshape(-1,1)
        term1+= xi**2
        term2+= np.cos(c*xi)
    term1= np.sqrt(1/D*term1)
    result= -a*np.exp(-b*term1) - np.exp(1/D*term2)+ a + np.exp(1)
    result = np.exp(-result / 10) +0.5
    return result

def ackley_cost_opt(D):

    y_opt= 0.0
    x_opt= np.zeros([1, D])
    domain=  [[-10,10]]*D

    return y_opt, x_opt, domain


def scipy_minimize_ackley_cost(D):

    def fun(x):
        x= x.reshape(1,-1)

        D = x.shape[1]
        term1 = 0
        term2 = 0
        a = 20
        b = 0.2
        c = np.pi * 2

        for i in range(D):
            xi = x[:, i].reshape(-1, 1)
            term1 += xi ** 2
            term2 += np.cos(c * xi)
        term1 = np.sqrt(1 / D * term1)
        result = -a * np.exp(-b * term1) - np.exp(1 / D * term2) + a + np.exp(1)
        result=  np.exp(-result/10) + 0.5
        result= result.flatten()

        return result

    domain=  [[-10,10]]*D

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


def ackley_cost_plots(disc, D, plot= False):

    domain=  [[-10,10]]*D
    if D==1:
        X_grid = get_grid(domain, disc)

    elif D==2:
        X_grid = get_grid(domain, disc)

    else:
        X_grid = get_grid(domain, disc)

    Y= ackley_cost(X_grid)

    if D==2:
        if plot:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_title('ackley')
            ax.scatter3D(X_grid[:,0], X_grid[:,1], Y[:,0], color= 'blue', label= 'posterior target')
            plt.show()
    return X_grid, Y

def ackley_cost_find_best_suited_kernel(X, Y, noise=10**(-4)):

    '''constraint values'''
    lower= 10**(-5); upper= 10**(6); #lengtscale and variance constarint
    lower_noise= 10**(-5); upper_noise= 10**(6); #noise constarint

    logistic = logistic_bjt(lower, upper)
    logistic_noise = logistic_bjt(lower_noise, upper_noise)

    D= X.shape[1]
    kernel = gp.kernels.RBF(lengthscales=np.array([1] * D))
    Y_latent= np.log(Y)
    model = gp.models.GPR((X, Y_latent), kernel=kernel)
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