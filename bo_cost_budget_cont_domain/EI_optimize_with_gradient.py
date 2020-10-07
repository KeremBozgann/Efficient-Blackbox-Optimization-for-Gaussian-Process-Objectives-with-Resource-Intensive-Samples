
import numpy as np
from Acquisitions import EI
import sys
sys.path.append('../')
from gp_gradients import dEI_dx

from scipy.optimize import minimize

import gpflow as gp
from scipy.optimize import Bounds
import  matplotlib.pyplot as plt



def EI_optimize_with_gradient(domain, model, random_restarts, num_iter_max, Xt, Yt, f_best, kernel, noise, D, grid):


    def EI_negative(x):

        x = x.reshape(1, -1)

        u_x, var_x = model.predict_f(x); u_x= u_x.numpy(); var_x= var_x.numpy()
        sigma_x = np.sqrt(var_x)
        EI_x= EI(sigma_x, u_x, f_best)

        EI_x= EI_x.flatten()
        return -EI_x


    def EI_negative_gradient(x):

        x= x.reshape(1,-1)
        u_x, var_x = model.predict_f(x); u_x = u_x.numpy(); var_x = var_x.numpy()
        sigma_x = np.sqrt(var_x)

        grad_x= dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise)

        grad_x= grad_x.flatten()
        return -grad_x

    if random_restarts>0:
        lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

        x_opt_list_sci= np.zeros([random_restarts, D])
        x_opt_value_list_sci= np.zeros([random_restarts, 1])

        low= []; upp= []
        for i in range(len(domain)):
            low.append(domain[i][0])
            upp.append(domain[i][1])
        b= Bounds(lb= low, ub= upp )

        for i in range(random_restarts):

            x0= np.random.uniform(lower, upper, (1,D))

            result= minimize(EI_negative, x0, bounds=b,  method='L-BFGS-B',
                              jac=EI_negative_gradient, options= {'maxiter': num_iter_max})

            x_opt_sci= result['x']; x_opt_value_sci= -result['fun']; x_opt_grad_sci= -result['jac']
            x_opt_sci= x_opt_sci.reshape(1,-1); x_opt_value_sci= x_opt_value_sci.reshape(1,-1); x_opt_grad_sci= x_opt_grad_sci.reshape(1,-1)

            x_opt_list_sci[i, :]= x_opt_sci[0, :]; x_opt_value_list_sci[i, :]= x_opt_value_sci[0,:];

        index_opt= int(np.argmax(x_opt_value_list_sci, axis=0))

        x_opt_value_sci= x_opt_value_list_sci[index_opt, :].reshape(1,-1)

        x_opt_sci= x_opt_list_sci[index_opt, :].reshape(1,-1)


    if D==1 and grid:
        disc = 100
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        X_grid = x1.reshape(-1, 1)

        uX_grid, varX_grid = model.predict_f(X_grid);
        uX_grid = uX_grid.numpy(); varX_grid = varX_grid.numpy()
        sigmaX_grid = np.sqrt(varX_grid)

        EIX_grid = EI(sigmaX_grid, uX_grid, f_best)

        index_max_grid= int(np.argmax(EIX_grid, axis=0))
        x_opt_value_grid= EIX_grid[index_max_grid, :].reshape(1,-1)
        x_opt_grid= X_grid[index_max_grid, :].reshape(1,-1)

    if D==2 and grid:
        disc= 21
        x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
        x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
        x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
        X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

        X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
        X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
        X_grid = np.append(X1_flat, X2_flat, axis=1)

        uX_grid, varX_grid = model.predict_f(X_grid);
        uX_grid = uX_grid.numpy(); varX_grid = varX_grid.numpy()
        sigmaX_grid = np.sqrt(varX_grid)

        EIX_grid = EI(sigmaX_grid, uX_grid, f_best)

        index_max_grid= int(np.argmax(EIX_grid, axis=0))
        x_opt_value_grid= EIX_grid[index_max_grid, :].reshape(1,-1)
        x_opt_grid= X_grid[index_max_grid, :].reshape(1,-1)

    if grid and (D==2 or D==1) and (random_restarts>0):
        if x_opt_value_grid> x_opt_value_sci:
            x_opt= x_opt_grid; x_opt_value= x_opt_value_grid

        else:
            x_opt= x_opt_sci; x_opt_value= x_opt_value_sci

    elif (random_restarts>0):
        x_opt= x_opt_sci; x_opt_value= x_opt_value_sci

    else:
        x_opt = x_opt_grid; x_opt_value = x_opt_value_grid

    return x_opt, x_opt_value

sys.path.append('../functions')

from sine import sin
import time
def test_EI_optimize_with_grad():

    domain= [[-2, 2]]; random_restarts= 5; D=1; noise= 10**(-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    x_opt_list= np.zeros([random_restarts, D])
    x_opt_value_list= np.zeros([random_restarts, 1])
    Xt= np.random.uniform(lower, upper, (10,D))
    Yt= sin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)


    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    t1 = time.clock()
    x_opt, x_opt_value= EI_optimize_with_gradient(domain, model, random_restarts,
                                          num_iter_max, Xt, Yt, f_best, kernel, noise, D)

    t2= time.clock()
    print('time:{}, num_iter_max:{}, random_restarts:{}'.format((t2- t1), num_iter_max, random_restarts))
    '''compare with grid'''
    disc = 100
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    X = x1.reshape(-1, 1)
    Y = sin(X)

    uX, varX = model.predict_f(X); uX = uX.numpy(); varX = varX.numpy()
    sigmaX = np.sqrt(varX)

    EIX= EI(sigmaX, uX, f_best)

    plt.figure()
    plt.plot(X[:,0], EIX[:,0], color= 'red')
    plt.scatter(X[:,0], EIX[:,0], color= 'red')
    plt.scatter(x_opt[0,0], x_opt_value[0,0], color= 'blue')
    plt.show()
