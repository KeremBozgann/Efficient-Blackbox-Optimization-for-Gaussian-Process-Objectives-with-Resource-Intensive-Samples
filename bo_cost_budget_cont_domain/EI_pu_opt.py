
import numpy as np
from Acquisitions import EI
from Acquisitions import EI_pu
import sys
sys.path.append('../')
from gp_gradients import dEI_dx

from scipy.optimize import minimize

import gpflow as gp
from scipy.optimize import Bounds
import  matplotlib.pyplot as plt

from gp_gradients import *

def EI_pu_optimize_with_gradient(domain, model, latent_cost_model, random_restarts, num_iter_max, Xt, Yt, Yt_cost, f_best,
                                 kernel, latent_cost_kernel, noise, D, grid):


    def EI_pu_negative(x):

        x = x.reshape(1, -1)

        u_x, var_x = model.predict_f(x); u_x= u_x.numpy(); var_x= var_x.numpy()
        sigma_x = np.sqrt(var_x)

        u_x_cost_latent, var_x_cost_latent = latent_cost_model.predict_f(x);
        u_x_cost_latent = u_x_cost_latent.numpy(); var_x_cost_latent = var_x_cost_latent.numpy()
        u_x_cost= np.exp(u_x_cost_latent)

        EI_x= EI(sigma_x, u_x, f_best)
        EI_pu_x= EI_x/u_x_cost

        EI_pu_x= EI_pu_x.flatten()
        return -EI_pu_x


    def EI_pu_negative_gradient(x):

        x= x.reshape(1,-1)
        u_x, var_x = model.predict_f(x); u_x = u_x.numpy(); var_x = var_x.numpy()
        sigma_x = np.sqrt(var_x)

        '''gradient of EI term'''
        grad_nom= dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise)

        '''gradient of denominator'''
        u_x_latent_cost, var_x_latent_cost = latent_cost_model.predict_f(x);
        u_x_latent_cost = u_x_latent_cost.numpy();
        var_x_latent_cost = var_x_latent_cost.numpy()
        u_x_cost = np.exp(u_x_latent_cost)

        Yt_latent_cost= np.log(Yt_cost)
        du_x_lat, dvar_x_lat, _ = mean_variance_gradients(latent_cost_kernel, Xt, Yt_latent_cost, x, noise)

        du_x_cost = du_x_lat * u_x_cost

        grad_denom= du_x_cost

        '''overall gradient'''
        ei_x= EI(sigma_x, u_x, f_best)
        grad_x= (u_x_cost*grad_nom- grad_denom* ei_x)/(u_x_cost**2)

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

            result= minimize(EI_pu_negative, x0, bounds=b,  method='L-BFGS-B',
                              jac=EI_pu_negative_gradient, options= {'maxiter': num_iter_max})

            x_opt_sci= result['x']; x_opt_value_sci= -result['fun']; x_opt_grad_sci= -result['jac']
            x_opt_sci= x_opt_sci.reshape(1,-1); x_opt_value_sci= x_opt_value_sci.reshape(1,-1); x_opt_grad_sci= x_opt_grad_sci.reshape(1,-1)

            x_opt_list_sci[i, :]= x_opt_sci[0, :]; x_opt_value_list_sci[i, :]= x_opt_value_sci[0,:];

        index_opt= int(np.argmax(x_opt_value_list_sci, axis=0))

        x_opt_value_sci= x_opt_value_list_sci[index_opt, :].reshape(1,-1)

        x_opt_sci= x_opt_list_sci[index_opt, :].reshape(1,-1)


    if D==1 and grid:
        disc = 101
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        X_grid = x1.reshape(-1, 1)

        uX_grid, varX_grid = model.predict_f(X_grid);
        uX_grid = uX_grid.numpy(); varX_grid = varX_grid.numpy()
        sigmaX_grid = np.sqrt(varX_grid)

        u_X_latent_cost_grid, var_X_latent_cost_grid = latent_cost_model.predict_f(X_grid);
        u_X_latent_cost_grid = u_X_latent_cost_grid.numpy();
        var_X_latent_cost_grid = var_X_latent_cost_grid.numpy()
        u_X_cost_grid = np.exp(u_X_latent_cost_grid)

        EI_pu_X_grid = EI_pu(sigmaX_grid, uX_grid, f_best, u_X_cost_grid)

        index_max_grid= int(np.argmax(EI_pu_X_grid, axis=0))
        x_opt_value_grid= EI_pu_X_grid[index_max_grid, :].reshape(1,-1)
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

        u_X_latent_cost_grid, var_X_latent_cost_grid = latent_cost_model.predict_f(X_grid);
        u_X_latent_cost_grid = u_X_latent_cost_grid.numpy();
        var_X_latent_cost_grid = var_X_latent_cost_grid.numpy()
        u_X_cost_grid = np.exp(u_X_latent_cost_grid)

        EI_pu_X_grid = EI_pu(sigmaX_grid, uX_grid, f_best, u_X_cost_grid)

        index_max_grid = int(np.argmax(EI_pu_X_grid, axis=0))
        x_opt_value_grid = EI_pu_X_grid[index_max_grid, :].reshape(1, -1)
        x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)


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
sys.path.append('../cost_functions')
from cos_1d import *
from exp_cos_1d import *

def temp_test_EI_pu_gradient():

    domain= [[-2, 2]]; random_restarts= 10; D=1; noise= 10**(-4); noise_cost= 10**(-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    x_opt_list= np.zeros([random_restarts, D])
    x_opt_value_list= np.zeros([random_restarts, 1])
    Xt= np.random.uniform(lower, upper, (10,D))
    Yt= sin(Xt)
    Yt_cost= exp_cos_1d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    grid= False

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    x_test= np.random.uniform(lower, upper, (1,D))
    x_test_val = -EI_pu_negative(x_test)
    grad_x_test= - EI_pu_negative_gradient(x_test)
    x_pl = x_test + 0.00001; x_mn = x_test - 0.00001;

    x_pl_val= -EI_pu_negative(x_pl)
    x_mn_val= -EI_pu_negative(x_mn)

    app_deriv= (( x_pl_val- x_test_val )/(x_pl- x_test)+\
                                    (x_test_val- x_mn_val)/(x_test- x_mn))/2

    print('\nanalytical_deriv:{}, app_deriv:{}'.format(grad_x_test, app_deriv))

def test_EI_pu_optimize_with_grad():

    domain= [[-2, 2]]; random_restarts= 10; D=1; noise= 10**(-4); noise_cost= 10**(-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    x_opt_list= np.zeros([random_restarts, D])
    x_opt_value_list= np.zeros([random_restarts, 1])
    Xt= np.random.uniform(lower, upper, (10,D))
    Yt= sin(Xt)
    Yt_cost= exp_cos_1d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    grid= False

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    t1 = time.clock()
    x_opt, x_opt_value= EI_pu_optimize_with_gradient(domain, model, latent_cost_model, random_restarts, num_iter_max, Xt,
                                        Yt, Yt_cost, f_best, kernel, latent_cost_kernel, noise, D, grid)


    t2= time.clock()
    print('time:{}, num_iter_max:{}, random_restarts:{}'.format((t2- t1), num_iter_max, random_restarts))
    '''compare with grid'''
    disc = 100
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    X = x1.reshape(-1, 1)
    Y = sin(X)

    uX, varX = model.predict_f(X); uX = uX.numpy(); varX = varX.numpy()
    sigmaX = np.sqrt(varX)

    u_X_latent_cost_grid, var_X_latent_cost_grid = latent_cost_model.predict_f(X);
    u_X_latent_cost_grid = u_X_latent_cost_grid.numpy();
    var_X_latent_cost_grid = var_X_latent_cost_grid.numpy()
    u_X_cost_grid = np.exp(u_X_latent_cost_grid)

    EIX_pu= EI_pu(sigmaX, uX, f_best, u_X_cost_grid)

    plt.figure()
    plt.plot(X[:,0], EIX_pu[:,0], color= 'red')
    plt.scatter(X[:,0], EIX_pu[:,0], color= 'red')
    plt.scatter(x_opt[0,0], x_opt_value[0,0], color= 'blue')
    plt.show()

from branin import branin
import time
sys.path.append('../cost_functions')
from cos_2d import *
from exp_cos_2d import *
from mpl_toolkits.mplot3d import Axes3D

def test_EI_pu_optimize_with_grad_branin():

    domain =[[-5,10], [0,15]]; random_restarts= 10; D=2; noise= 10**(-4); noise_cost= 10**(-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (10,D))
    Yt= branin(Xt)
    Yt_cost= exp_cos_2d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    grid= False

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    x_opt, x_opt_value= EI_pu_optimize_with_gradient(domain, model, latent_cost_model, random_restarts, num_iter_max, Xt,
                                        Yt, Yt_cost, f_best, kernel, latent_cost_kernel, noise, D, grid)


    print('time:{}, num_iter_max:{}, random_restarts:{}'.format((t2- t1), num_iter_max, random_restarts))
    '''compare with grid'''
    disc = 21
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    uX, varX = model.predict_f(X); uX = uX.numpy(); varX = varX.numpy()
    sigmaX = np.sqrt(varX)

    u_X_latent_cost_grid, var_X_latent_cost_grid = latent_cost_model.predict_f(X);
    u_X_latent_cost_grid = u_X_latent_cost_grid.numpy();
    var_X_latent_cost_grid = var_X_latent_cost_grid.numpy()
    u_X_cost_grid = np.exp(u_X_latent_cost_grid)

    EIX_pu= EI_pu(sigmaX, uX, f_best, u_X_cost_grid)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter3D(X[:,0], X[:,1], EIX_pu[:,0], color= 'red', alpha= 0.5)

    ax.scatter(x_opt[0,0], x_opt[0,1], x_opt_value[0,0], marker= 'X' ,color= 'blue')
    plt.show()
