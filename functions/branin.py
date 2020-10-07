import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gpflow as gp
from gpflow.utilities import print_summary
import sys
sys.path.append('../bo_cost_budget_cont_domain')
from scipy.optimize import minimize, Bounds
from hyperparameter_optimization import logistic_bjt

def branin(X):

    x1= X[:,0].reshape(-1,1); x2= X[:,1].reshape(-1,1)
    result = np.square(x2 - (5.1 / (4 * np.square(np.pi))) * np.square(x1) +
                       (5 / np.pi) * x1 - 6) + 10 * (1 - (1. / (8 * np.pi))) * np.cos(x1) + 10

    # result = float(result)

    # print('Result = %f' % result)
    # time.sleep(np.random.randint(60))
    return result

def branin_opt():

    y_opt= 0.397887
    x_opt= [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    domain= [[-5,10], [0,15]]

    return y_opt, x_opt, domain


def scipy_minimize_branin():

    def fun(x):
        x= x.reshape(1,-1)
        x1 = x[:, 0].reshape(-1, 1);
        x2 = x[:, 1].reshape(-1, 1)
        result = np.square(x2 - (5.1 / (4 * np.square(np.pi))) * np.square(x1) +
                           (5 / np.pi) * x1 - 6) + 10 * (1 - (1. / (8 * np.pi))) * np.cos(x1) + 10
        result= result.flatten()
        return result

    domain= [[-5,10], [0,15]]

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



def branin_plots(disc, plot= False):



    domain =[[-5,10], [0,15]]
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    Y= branin(X)

    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('branin')
        ax.scatter3D(X[:,0], X[:,1], Y[:,0], color= 'blue', label= 'posterior target')
        plt.show()
    return X, Y

def branin_find_best_suited_kernel(X, Y, noise=10**(-4)):

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