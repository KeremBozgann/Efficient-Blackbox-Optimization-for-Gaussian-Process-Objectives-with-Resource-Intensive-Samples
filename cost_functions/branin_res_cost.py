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

def branin_res_cost(X):

    x1= X[:,0].reshape(-1,1); x2= X[:,1].reshape(-1,1)

    x1_res = 15 * x1 - 5;x2_res = 15 * x2
    res= 1/51.95
    result = res*(np.square(x2_res - (5.1 / (4 * np.square(math.pi))) * np.square(x1_res) +
                       (5 / math.pi) * x1_res - 6) + 10 * (1 - (1. / (8 * math.pi))) * np.cos(x1_res) -44.81)

    # result = float(result)

    # print('Result = %f' % result)
    # time.sleep(np.random.randint(60))
    return np.exp(-result)+0.5

def branin_res_cost_opt():

    y_opt= -1.04739389
    x_opt= [[0.12389382, 0.81833334], [0.96165187, 0.165], [0.54277284, 0.15166667]]
    domain= [[0,1], [0,1]]

    return y_opt, x_opt, domain


def scipy_minimize_branin_res_cost():

    def fun(x):
        x= x.reshape(1,-1)
        x1 = x[:, 0].reshape(-1, 1);
        x2 = x[:, 1].reshape(-1, 1)

        x1_res = 15 * x1 - 5;
        x2_res = 15 * x2
        res = 1 / 51.95
        result = res * (np.square(x2_res - (5.1 / (4 * np.square(math.pi))) * np.square(x1_res) +
                                  (5 / math.pi) * x1_res - 6) + 10 * (1 - (1. / (8 * math.pi))) * np.cos(x1_res) - 44.81)

        return -np.exp(-result)+0.5

    domain= [[0,1], [0,1]]

    lower = [];upper = []
    D = 2

    for i in range(len(domain)):
        lower.append(domain[i][0]); upper.append(domain[i][1])
    b = Bounds(lb=lower, ub=upper)

    x_opt_list= np.empty([100, 2])
    value_list= np.empty([100, 1])

    for i in range(100):

        x0 = np.random.uniform(lower, upper, (1, D))
        result= minimize(fun= fun, bounds= b, x0= x0, method= 'L-BFGS-B')
        # print('optimum point:{}, optimum value:{}'.format(result['x'], result['fun']))
        x_opt_list[i, :] = result['x'].reshape(1,-1)
        value_list[i, :]= result['fun'].reshape(1,-1)
    index= np.argmin(value_list, axis=0)
    print('optimum value:{}, optimum point:{}'.format(value_list[index, :].reshape(-1,1), x_opt_list[index , :].reshape(1,-1)))


def branin_res_cost_plots(disc, plot= False):



    domain= [[0,1], [0,1]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    Y= branin_res_cost(X)

    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('branin')
        ax.scatter3D(X[:,0], X[:,1], Y[:,0], color= 'blue', label= 'posterior target')
        plt.show()
    return X, Y

def branin_res_cost_find_best_suited_kernel(X, Y, noise=10**(-4)):

    '''constraint values'''
    lower= 10**(-5); upper= 10**(6); #lengtscale and variance constarint
    lower_noise= 10**(-5); upper_noise= 10**(6); #noise constarint

    logistic = logistic_bjt(lower, upper)
    logistic_noise = logistic_bjt(lower_noise, upper_noise)

    D= X.shape[1]
    kernel = gp.kernels.RBF(lengthscales=np.array([1] * D))
    Y_lat=np.log(Y)
    model = gp.models.GPR((X, Y_lat), kernel=kernel)
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