
import numpy as np
import gpflow as gp
import tensorflow as tf

import sys
sys.path.append('../')
from Acquisitions import EI
from Acquisitions import EI_pu

from scipy.stats import norm
from scipy.optimize import minimize
import  matplotlib.pyplot as plt
from scipy.optimize import Bounds

def mean_variance_gradients(kernel, Xt, Yt, x, noise):

    '''implement du_dx and gradient of K(x)= k(x,x)- k(x,Xt)(K(Xt,Xt)+I*noise)_inv*k(x,Xt)_T'''
    # K_inv = np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)

    k_x_x= kernel.K(x,x); k_x_x= k_x_x.numpy()
    k_x_Xt= kernel.K(x,Xt); k_x_Xt= k_x_Xt.numpy()
    k_x_Xt_T= np.transpose(k_x_Xt)
    k_Xt_Xt= (kernel.K(Xt,Xt)).numpy(); noise_mat= np.eye(Xt.shape[0])*noise

    K_inv= np.linalg.inv(k_Xt_Xt+noise_mat)

    K_x= k_x_x- np.matmul(k_x_Xt, np.matmul(K_inv, k_x_Xt_T))

    alpha= np.matmul(K_inv,Yt)

    ls = (kernel.lengthscales._value()).numpy()

    if type(ls)== np.ndarray:

        lamb= np.diagflat(ls)
        lamb_inv= np.diagflat(1/ls)
    else:

        lamb= np.eye(Xt.shape[1])*ls
        lamb_inv= np.eye(Xt.shape[1])*1/ls

    X_vec= x- Xt

    # du_dx= np.matmul(np.matmul(-lamb_inv, X_vec.T), kernel.K(Xt, x)* alpha)
    du_dx= np.matmul(np.matmul(-lamb_inv, X_vec.T), k_x_Xt_T* alpha)


    '''variance'''
    # gradient_vec= np.matmul(2*lamb_inv, ((x.T-Xt.T)*(kernel.K(x, Xt).numpy())));
    temp1= np.matmul(2*lamb_inv, ((x.T-Xt.T)*(k_x_Xt)));
    temp2= np.matmul(temp1, K_inv);
    # dvar_dx= np.matmul(temp1, kernel.K(Xt, x))
    dvar_dx= np.matmul(temp2, k_x_Xt_T)


    return du_dx, dvar_dx, K_x


def temp_test_mean_variance_gradients(kernel, Xt, Yt, noise, model):

    x_test= np.array([[-1.0]]); x_plus= x_test+ 0.0001; x_minus= x_test- 0.001
    du_dx_test, dvar_dx_test, K_x_test= mean_variance_gradients(kernel, Xt, Yt, x_test, noise)

    u0_x_test, var0_x_test= model.predict_f(x_test); u0_x_test= u0_x_test.numpy(); var0_x_test= var0_x_test.numpy()
    u0_x_plus, var0_x_plus= model.predict_f(x_plus); u0_x_plus= u0_x_plus.numpy(); var0_x_plus= var0_x_plus.numpy()
    u0_x_minus, var0_x_minus= model.predict_f(x_minus); u0_x_minus= u0_x_minus.numpy(); var0_x_minus= var0_x_minus.numpy()
    app_du_dx_test=  ((u0_x_plus- u0_x_test)/(x_plus- x_test)+ (u0_x_test- u0_x_minus)/(x_test- x_minus))/2

    print('du_dx_test:{},app_du_dx_test:{}'.format(du_dx_test, app_du_dx_test))

    app_dvar_dx_test=   ((var0_x_plus- var0_x_test)/(x_plus- x_test)+ (var0_x_test- var0_x_minus)/(x_test- x_minus))/2

    print('dvar_dx_test:{},app_dvar_dx_test:{}'.format(dvar_dx_test, app_dvar_dx_test))


def dcumulative_du(f_best, sigma_x, u_x):

    return -1/sigma_x*norm.pdf((f_best- u_x)/sigma_x)

def dcumulative_dsigma(f_best, sigma_x, u_x):

    return -(f_best-u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)

def dcumulative_dfbest(f_best, sigma_x, u_x):

    return 1/sigma_x*norm.pdf((f_best- u_x)/sigma_x)

def dnormal_du(f_best, u_x, sigma_x):

    return (f_best- u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)

def dnormal_dsigma(f_best, u_x, sigma_x):


    return ((f_best-u_x)**2)/(sigma_x**3)*norm.pdf((f_best- u_x)/sigma_x)

def dnormal_dfbest(f_best, u_x, sigma_x):

    return -(f_best - u_x) / (sigma_x** 2) * norm.pdf((f_best - u_x) / sigma_x)

def dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise):

    du_dx, dvar_dx, _= mean_variance_gradients(kernel, Xt, Yt, x, noise)

    dsigma_dx= (1/(2*sigma_x))*dvar_dx

    '''cumulative gradient'''
    dcumulative_dx= dcumulative_du(f_best, sigma_x,u_x)*du_dx + \
                            dcumulative_dsigma(f_best, sigma_x, u_x)*dsigma_dx

    term1= (f_best- u_x)*dcumulative_dx

    term2= -du_dx* norm.cdf(((f_best- u_x)/sigma_x))

    term3= dsigma_dx* norm.pdf((f_best- u_x)/sigma_x)

    '''normal distribution gradient'''

    dnormal_dx= dnormal_du(f_best, u_x, sigma_x)*du_dx+ dnormal_dsigma(f_best, u_x, sigma_x)*dsigma_dx

    term4= sigma_x* dnormal_dx

    return term1+ term2+ term3+ term4


def posterior_covariance(x, x1, Xt, kernel, noise):

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    temp= np.matmul((kernel.K(x,Xt)).numpy(), K_inv)
    result= (kernel.K(x,x1)).numpy()- np.matmul(temp, ((kernel.K(x1, Xt)).numpy()).T)

    return result

def one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise):

    if type(Z) != np.ndarray:
        Z = np.atleast_2d(Z)

    k0_x2_x1= posterior_covariance(x2, x1, Xt, kernel,  noise)
    k0_x1_x1= posterior_covariance(x1, x1, Xt, kernel, noise)
    L0_x1= np.linalg.cholesky(k0_x1_x1)
    sigma0= np.matmul(k0_x2_x1, np.linalg.inv(L0_x1))

    u1_x2= u0_x2 + np.matmul(sigma0,Z)

    var1_x2= var0_x2- np.matmul(sigma0, sigma0.T)


    return u1_x2, var1_x2

def one_step_f_best(f_best, u_x1, var_x1, Z):

    if type(Z) != np.ndarray:
        Z = np.atleast_2d(Z)

    L0_x1= np.linalg.cholesky(var_x1)

    f_x1= u_x1+ np.matmul(L0_x1,Z)

    f_x1min= np.min(f_x1)
    f_best1= np.minimum(f_best, float(f_x1min))

    return f_best1


def EI1_x2_pu_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, latent_cost_model, f_best):

    '''observation at x1'''

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    sigma0_x1= np.sqrt(var0_x1)
    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)
    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    # '''update model according to outcome of Z'''

    model1= gp.models.GPR((Xt1, Yt1), kernel)
    model1.likelihood.variance.assign(noise)


    u0_x1, var0_x1 = model.predict_f(x1); u0_x1 = u0_x1.numpy(); var0_x1 = var0_x1.numpy()


    '''update f_best according to the outcome of Z'''

    f_best1 = one_step_f_best(f_best, u0_x1, var0_x1, Z_conv)

    u1_X2_grid, var1_X2_grid = model1.predict_f(X2_grid); u1_X2_grid= u1_X2_grid.numpy(); var1_X2_grid= var1_X2_grid.numpy()
    sigma1_X2_grid = np.sqrt(var1_X2_grid)

    u0_latcost_X2_grid, var0_latcost_X2_grid = latent_cost_model.predict_f(X2_grid);
    u0_latcost_X2_grid= u0_latcost_X2_grid.numpy(); var0_latcost_X2_grid= var0_latcost_X2_grid.numpy()
    u0_cost_X2_grid= np.exp(u0_latcost_X2_grid)

    EI1_X2_grid= EI(sigma1_X2_grid, u1_X2_grid, f_best1)

    EI1_pu_X2_grid= EI1_X2_grid/u0_cost_X2_grid

    index_max_grid= int(np.argmax(EI1_pu_X2_grid, axis=0))
    x2_opt= X2_grid[index_max_grid, :].reshape(1,-1)
    x2_opt_value= EI1_pu_X2_grid[index_max_grid, :].reshape(1,-1)

    '''EI1_x2_opt'''
    ei1_x2_opt= EI1_X2_grid[index_max_grid, :].reshape(1,-1)
    return x2_opt, x2_opt_value, ei1_x2_opt

def EI1_x2_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, f_best):

    '''observation at x1'''

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    sigma0_x1= np.sqrt(var0_x1)
    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)
    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    # '''update model according to outcome of Z'''

    model1= gp.models.GPR((Xt1, Yt1), kernel)
    model1.likelihood.variance.assign(noise)


    u0_x1, var0_x1 = model.predict_f(x1); u0_x1 = u0_x1.numpy(); var0_x1 = var0_x1.numpy()

    '''update f_best according to the outcome of Z'''

    f_best1 = one_step_f_best(f_best, u0_x1, var0_x1, Z_conv)

    u1_X2_grid, var1_X2_grid = model1.predict_f(X2_grid); u1_X2_grid= u1_X2_grid.numpy(); var1_X2_grid= var1_X2_grid.numpy()

    sigma1_X2_grid= np.sqrt(var1_X2_grid)

    EI1_X2_grid= EI(sigma1_X2_grid, u1_X2_grid, f_best1)

    index_max_grid= int(np.argmax(EI1_X2_grid, axis=0))
    x2_opt= X2_grid[index_max_grid, :].reshape(1,-1)
    x2_opt_value= EI1_X2_grid[index_max_grid, :].reshape(1,-1)

    return x2_opt, x2_opt_value

def  EI1_x2_optimize_grid_with_for(X2_grid, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, f_best):


    fbest1 = one_step_f_best(f_best, u0_x1, var0_x1, Z)

    EI_grid = np.empty([X2_grid.shape[0], 1])

    for i in range(X2_grid.shape[0]):
        x2i = X2_grid[i, :].reshape(1, -1)
        u0_x2i, var0_x2i = model.predict_f(x2i);
        u0_x2i = u0_x2i.numpy();
        var0_x2i = var0_x2i.numpy()

        u1_x2i, var1_x2i = one_step_mean_and_variance(x2i, u0_x2i, var0_x2i, kernel, x1, Z, Xt, noise)
        sigma1_x2i = np.sqrt(var1_x2i)
        EIi = EI(sigma1_x2i, u1_x2i, fbest1)
        EI_grid[i, 0] = EIi

    index_opt = int(np.argmax(EI_grid, axis=0))
    x2_opt_grid_for = X2_grid[index_opt, :].reshape(1, -1)
    x2_opt_value_grid_for = EI_grid[index_opt, :].reshape(1,-1)

    return x2_opt_grid_for, x2_opt_value_grid_for

import time
def temp_test_EI1_x2_optimize_grid(Xt, Yt, f_best, kernel, model, domain, noise, Z_test, D):

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]
    x1_test = np.random.uniform(lower, upper, (1, D))

    u0_x1, var0_x1 = model.predict_f(x1_test);
    u0_x1 = u0_x1.numpy();
    var0_x1 = var0_x1.numpy()
    sigma0_x1 = np.sqrt(var0_x1)



    disc = 101
    x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
    X2_grid = x1_grid.reshape(-1, 1)


    time_gp1= time.clock()
    x2_opt_grid, x2_opt_value_grid = EI1_x2_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1_test, Z_test,
                                                          Xt, Yt, noise, model, f_best)

    time_gp2 = time.clock()
    time_gp = time_gp2- time_gp1

    # u0_x1_test, var0_x1_test = model.predict_f(x1_test);
    # u0_x1_test = u0_x1_test.numpy();
    # var0_x1_test = var0_x1_test.numpy()
    # L0_x1_test= np.linalg.cholesky(var0_x1_test)
    # L0_x1_test_inv= np.linalg.inv(L0_x1_test)


    '''find EI using for loop and compare'''

    time_for_1= time.clock()

    fbest1 = one_step_f_best(f_best, u0_x1, var0_x1, Z_test)

    EI_grid= np.empty([X2_grid.shape[0], 1])

    for i in range(X2_grid.shape[0]):

        x2i= X2_grid[i,:].reshape(1,-1)
        u0_x2i, var0_x2i = model.predict_f(x2i); u0_x2i = u0_x2i.numpy();
        var0_x2i = var0_x2i.numpy()

        u1_x2i, var1_x2i= one_step_mean_and_variance(x2i, u0_x2i, var0_x2i, kernel, x1_test, Z_test, Xt, noise)
        sigma1_x2i= np.sqrt(var1_x2i)
        EIi= EI(sigma1_x2i, u1_x2i, fbest1)
        EI_grid[i, 0] = EIi
    index_opt= int(np.argmax(EI_grid, axis= 0))
    x2_opt_grid_other= X2_grid[index_opt, :].reshape(1,-1)
    x2_opt_value_grid_other= EI_grid[index_opt, 0]

    time_for_2= time.clock()
    time_for= time_for_2- time_for_1

    print('x2_opt_grid:{}, x2_opt_grid_other:{}'.format(x2_opt_grid, x2_opt_grid_other))
    print('x2_opt_grid_value:{}, x2_opt_grid_value_other:{}'.format(x2_opt_value_grid, x2_opt_value_grid_other))
    print('time_gp:{}, time_for:{}'.format(time_gp, time_for))


def EI1_x2_per_cost_optimize(u0_x1, var0_x1, kernel, latent_cost_kernel, x1, Z, Xt, Yt, Yt_latent_cost,
                             noise, model, latent_cost_model, f_best, domain, num_inner_opt_restarts, grid_opt_in, D, f_best1, num_iter_max):


    '''observation at x1'''

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    sigma0_x1= np.sqrt(var0_x1)
    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)
    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)


    def EI_x2_pu_negative(x2):

        # print('x2 shape before reshaping in obj', x2.shape)
        x2= x2.reshape(1,-1)
        # print('x2 shape in obj', x2.shape)
        u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

        u0_x2_latent_cost, var0_x2_latent_cost= latent_cost_model.predict_f(x2); u0_x2_latent_cost= u0_x2_latent_cost.numpy();
        var0_x2_latent_cost= var0_x2_latent_cost.numpy()
        u0_x2_cost= np.exp(u0_x2_latent_cost)

        u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)

        sigma1_x2= np.sqrt(var1_x2)

        EI1_x2_pu= EI_pu(sigma1_x2, u1_x2 , f_best1, u0_x2_cost)

        # print('EI1 shape before reshaping in obj', EI1_x2.shape)
        EI1_x2_pu= EI1_x2_pu.flatten()
        # print('ei1 shape in obj',EI1_x2.shape)
        return -EI1_x2_pu


    def grad_EI_X2_pu_negative(x2):

        '''dEI1_x2'''
        x2 = x2.reshape(1, -1)

        u0_x2, var0_x2 = model.predict_f(x2); u0_x2 = u0_x2.numpy(); var0_x2 = var0_x2.numpy()

        u1_x2, var1_x2 = one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)
        sigma1_x2 = np.sqrt(var1_x2)

        EI1_x2_gradient = dEI_dx(f_best1, u1_x2, sigma1_x2, kernel, Xt1, Yt1, x2, noise)

        '''gradient of mean cost'''

        u0_x2_latent_cost, var0_x2_latent_cost= latent_cost_model.predict_f(x2); u0_x2_latent_cost= u0_x2_latent_cost.numpy();
        var0_x2_latent_cost= var0_x2_latent_cost.numpy()
        u0_x2_cost= np.exp(u0_x2_latent_cost)

        du0_x2_lat, dvar0_lat, _= mean_variance_gradients(latent_cost_kernel, Xt, Yt_latent_cost, x2, noise)

        du0_x2_cost= du0_x2_lat* u0_x2_cost

        '''EI1_x2'''

        EI1_x2 = EI(sigma1_x2, u1_x2, f_best1)

        '''overall gradient term'''
        grad_pu= (u0_x2_cost*EI1_x2_gradient- du0_x2_cost*EI1_x2)/(u0_x2_cost**2)
        grad_pu= grad_pu.flatten()

        return -grad_pu

    '''scipy optimize'''
    if num_inner_opt_restarts>0:
        lower= []; upper= []
        for i in range(len(domain)):
            lower.append(domain[i][0])
            upper.append(domain[i][1])
        b= Bounds(lb= lower, ub= upper )

        x2_opt_list_sci= np.zeros([num_inner_opt_restarts, D])
        x2_opt_value_list_sci= np.zeros([num_inner_opt_restarts, 1])

        for i in range(num_inner_opt_restarts):

            x20= np.random.uniform(lower, upper, (1, D))
            result= (minimize(EI_x2_pu_negative, x20, bounds=b,  method='L-BFGS-B', jac=grad_EI_X2_pu_negative,options={'maxiter':num_iter_max}))
            x2_cand= result['x']; x2_cand= x2_cand.reshape(1,-1)
            x2_cand_value= -result['fun']; x2_cand_value= x2_cand_value.reshape(1,-1)
            x2_cand_grad= -result['jac']; x2_cand_grad= x2_cand_grad.reshape(1,-1)

            x2_opt_list_sci[i, :]= x2_cand[0,:];
            x2_opt_value_list_sci[i, :]= x2_cand_value

        index_opt_sci= int(np.argmax(x2_opt_value_list_sci, axis=0))

        x2_opt_value_sci= x2_opt_value_list_sci[index_opt_sci, :].reshape(1,-1)

        x2_opt_sci= x2_opt_list_sci[index_opt_sci, :].reshape(1,-1)

        u0_x2_opt_sci, var0_x2_opt_sci = model.predict_f(x2_opt_sci); u0_x2_opt_sci = u0_x2_opt_sci.numpy();
        var0_x2_opt_sci = var0_x2_opt_sci.numpy()
        u1_x2, var1_x2= one_step_mean_and_variance(x2_opt_sci, u0_x2_opt_sci, var0_x2_opt_sci, kernel, x1, Z, Xt, noise)
        sigma1_x2= np.sqrt(var1_x2)
        ei1_x2_opt_sci = EI(sigma1_x2, u1_x2, f_best1)

    '''grid optimize'''

    '''inner optimization with grid'''
    if D == 1 and grid_opt_in:
        disc = 101
        x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
        X2_grid = x1_grid.reshape(-1, 1)

        x2_opt_grid, x2_opt_value_grid, ei1_x2_opt_grid =  EI1_x2_pu_optimize_grid(X2_grid, u0_x1, var0_x1, kernel,
                                                      x1, Z, Xt, Yt, noise, model, latent_cost_model, f_best)

    if D == 2 and grid_opt_in:
        disc = 21
        x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
        x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
        # x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
        X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

        X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
        X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
        X2_grid = np.append(X1_flat, X2_flat, axis=1)

        x2_opt_grid, x2_opt_value_grid, ei1_x2_opt_grid  =  EI1_x2_pu_optimize_grid(X2_grid, u0_x1, var0_x1, kernel,
                                                      x1, Z, Xt, Yt, noise, model, latent_cost_model, f_best)

    '''compare grid with sci optimum'''

    if (D == 2 or D == 1) and grid_opt_in and (num_inner_opt_restarts > 0):
        if x2_opt_value_grid > x2_opt_value_sci:
            x2_opt = x2_opt_grid;
            x2_opt_value = x2_opt_value_grid
            ei1_x2_opt= ei1_x2_opt_grid
        else:
            x2_opt = x2_opt_sci;
            x2_opt_value = x2_opt_value_sci
            ei1_x2_opt= ei1_x2_opt_sci
    elif num_inner_opt_restarts > 0:
        x2_opt = x2_opt_sci;
        x2_opt_value = x2_opt_value_sci
        ei1_x2_opt = ei1_x2_opt_sci
    else:
        x2_opt = x2_opt_grid
        x2_opt_value = x2_opt_value_grid
        ei1_x2_opt = ei1_x2_opt_grid

    return x2_opt_value, x2_opt, ei1_x2_opt

def EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, f_best, domain):


    '''observation at x1'''
    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    sigma0_x1= np.sqrt(var0_x1)
    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)
    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    # '''update model according to outcome of Z'''
    #
    # model1= gp.models.GPR((Xt1, Yt1), kernel)
    # model1.likelihood.variance.assign(noise)
    #


    # u0_x1, var0_x1 = model.predict_f(x1); u0_x1 = u0_x1.numpy(); var0_x1 = var0_x1.numpy()

    '''update f_best according to the outcome of Z'''

    f_best1 = one_step_f_best(f_best, u0_x1, var0_x1, Z_conv)

    def EI_x2(x2):

        # print('x2 shape before reshaping', x2.shape)
        x2= x2.reshape(1,-1)

        # print('x2 shape in obj', x2.shape)
        u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

        u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)

        sigma1_x2= np.sqrt(var1_x2)

        EI1_x2= EI(sigma1_x2, u1_x2, f_best1)

        # print('EI1 shape before reshaping', EI1_x2.shape)
        EI1_x2= EI1_x2.reshape(1,-1)

        EI1_x2= EI1_x2.flatten()
        return EI1_x2

    def EI_x2_negative(x2):

        # print('x2 shape before reshaping in obj', x2.shape)
        x2= x2.reshape(1,-1)
        # print('x2 shape in obj', x2.shape)
        u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

        u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)

        sigma1_x2= np.sqrt(var1_x2)

        EI1_x2= EI(sigma1_x2, u1_x2, f_best1)

        # print('EI1 shape before reshaping in obj', EI1_x2.shape)
        EI1_x2= EI1_x2.flatten()
        # print('ei1 shape in obj',EI1_x2.shape)
        return -EI1_x2

    def grad_EI_X2(x2):

        x2= x2.reshape(1,-1)

        u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

        u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)
        sigma1_x2= np.sqrt(var1_x2)


        EI1_x2_gradient= dEI_dx(f_best1, u1_x2, sigma1_x2, kernel, Xt1, Yt1, x2, noise)

        EI1_x2_gradient = EI1_x2_gradient.flatten()
        return EI1_x2_gradient

    def grad_EI_X2_negative(x2):
        # print('x2 shape in grad before reshaping', x2.shape)
        x2 = x2.reshape(1, -1)
        # print('x2 shape in grad', x2.shape)
        u0_x2, var0_x2 = model.predict_f(x2);
        u0_x2 = u0_x2.numpy();
        var0_x2 = var0_x2.numpy()

        u1_x2, var1_x2 = one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)
        sigma1_x2 = np.sqrt(var1_x2)

        EI1_x2_gradient = dEI_dx(f_best1, u1_x2, sigma1_x2, kernel, Xt1, Yt1, x2, noise)
        # print('ei_grad shape in grid before reshaping', EI1_x2_gradient.shape)
        EI1_x2_gradient = EI1_x2_gradient.flatten()
        # print('ei_grad shape in grad', EI1_x2_gradient.shape)
        return -EI1_x2_gradient


    lower= []; upper= []
    for i in range(len(domain)):
        lower.append(domain[i][0])
        upper.append(domain[i][1])
    b= Bounds(lb= lower, ub= upper )

    result= (minimize(EI_x2_negative, x20, bounds=b,  method='L-BFGS-B', jac=grad_EI_X2_negative))
    x2_opt_nonglobal= result['x'];
    opt_value= -result['fun']
    x2_opt_grad= -result['jac']

    x2_opt_nonglobal= x2_opt_nonglobal.reshape(1,-1)
    opt_value= opt_value.reshape(1,-1)
    x2_opt_grad= x2_opt_grad.reshape(1,-1)

    return x2_opt_nonglobal, opt_value, x2_opt_grad, result

# def test_gradient():

# def basic_gradient_descent_optimizer(x2, func, grad, alpha= 0.01):
#
#     grad= grad(x2)
#     func_value= func(x2)
#     x2_new= x2- grad(x2)

def temp(X2, u1_X2_model1, sigma1_X2_model1, f_best1, x2_opt, opt_value,ei_X2):

    EI1_X2_model1= EI(sigma1_X2_model1, u1_X2_model1, f_best1)
    plt.figure()
    plt.plot(X2[:,0], EI1_X2_model1[:,0], color= 'red' )
    plt.plot(X2, ei_X2, color= 'green' )
    plt.scatter(x2_opt.flatten(), opt_value)
    plt.show()
    # return EI1_X_model1

def temp_test_EI_x2_result():

    X2=np.linspace(-2,2, 101)
    X2= X2.reshape(-1,1)

    ei_X2= np.empty(X2.shape); u1_X2= np.empty(X2.shape); var1_X2= np.empty(X2.shape)

    for i, x2 in enumerate(X2):

        ei_X2[i,0]= EI_x2(x2)

    plt.figure()
    plt.plot(X2, ei_X2)
    plt.show()
    return ei_X2, X2

def test_mean_and_variance_obtained_analytically_and_with_model_update(x2, x1, model, Z):

    u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()
    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()

    L0_x1= np.linalg.cholesky(var0_x1); y1= u0_x1+ np.matmul(L0_x1,Z)

    '''analytical u1_x2, var1_x2'''
    u1_x2, var1_x2 = one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)
    sigma1_x2 = np.sqrt(u1_x2)


def posterior_covariance_and_derivative_1q(x, x1, Xt, kernel, noise):

    #K_x1_x= K_x_x1

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    k_x1_Xt= (kernel.K(x1,Xt)).numpy()
    k_x_Xt_T= np.transpose((kernel.K(x, Xt)).numpy())
    alpha= np.matmul(K_inv, k_x_Xt_T)
    k_x1_x= (kernel.K(x1,x)).numpy()
    K_x1_x= (k_x1_x- np.matmul(k_x1_Xt, alpha))


    '''gradient for 1d'''
    ls = (kernel.lengthscales._value()).numpy()

    if type(ls)== np.ndarray:

        lamb= np.diagflat(ls)
        lamb_inv= np.diagflat(1/ls)
    else:

        lamb= np.eye(Xt.shape[1])*ls
        lamb_inv= np.eye(Xt.shape[1])*1/ls

    k_x1_x= (kernel.K(x1,x)).numpy()
    diff_vec= (x-x1).T
    term1= np.matmul(lamb_inv, (diff_vec* k_x1_x))
    term2 = -np.matmul(lamb_inv, np.matmul(np.transpose(x1-Xt)*k_x1_Xt, alpha))
    dK_x1_x_dx1= term1-term2

    return K_x1_x, dK_x1_x_dx1

def funfi(A):

    result= np.tril(A, k=-1)+ 1/2*np.diagflat(np.diag(A))
    return result

def dsigma0_dx1_q1(x2_opt, x1, Xt, kernel, noise, K0_x1_x1,  dK0_x1_dx1, D, Yt):

    #K0_x1: 1*1, dK0_x1_dx1: D*1
    K0_x2_x1, dK0_x2_x1_dx1 = posterior_covariance_and_derivative_1q(x2_opt, x1, Xt, kernel, noise)
    # _, dK0_x1_dx1, K0_x1_x1= mean_variance_gradients(kernel, Xt, Yt, x1, noise)

    L0_x1= np.linalg.cholesky(K0_x1_x1)
    L0_x1_inv= np.linalg.inv(L0_x1); L0_x1_inv_T= np.transpose(L0_x1_inv)

    dL0_x1_dx1 = np.empty([D, 1])
    for i in range(D):
        dK0_x1_dx1i = dK0_x1_dx1[i, :].reshape(1, 1)
        Ai = np.matmul(L0_x1_inv, np.matmul(dK0_x1_dx1i, L0_x1_inv_T))
        temp = funfi(Ai)
        dL0_x1_dx1i = np.matmul(L0_x1, temp)
        dL0_x1_dx1[i, 0] = dL0_x1_dx1i

    '''only for q=1'''
    dL0_x1_inv_dx1= np.empty([D,1])
    for i in range(D):
        dL0_x1_dx1i= dL0_x1_dx1[i,:].reshape(1,1)
        dL0_x1_inv_dx1i= -np.matmul(L0_x1_inv, np.matmul(dL0_x1_dx1i, L0_x1_inv))
        dL0_x1_inv_dx1[i,0]= dL0_x1_inv_dx1i

    '''only for q=1'''
    dsigma0_dx1= np.empty([D,1])
    for i in range(D):
        dK0_x2_x1_dx1i= dK0_x2_x1_dx1[i,:].reshape(1,1);
        dL0_x1_inv_dx1i= dL0_x1_inv_dx1[i,:].reshape(1,1)
        dsigma0_dx1i= np.matmul(np.transpose(L0_x1_inv),dK0_x2_x1_dx1i)+ np.matmul(K0_x2_x1, dL0_x1_inv_dx1i)
        dsigma0_dx1[i,0]= dsigma0_dx1i

    sigma0= np.matmul(K0_x2_x1, L0_x1_inv)

    return dsigma0_dx1, sigma0, dL0_x1_dx1, L0_x1

def temp_test_dsigma0_dx1_q1(x2, x_test, Xt, kernel, noise, Yt, D, domain, model):

    x2= np.random.uniform(domain[0][0], domain[0][1], (1,1))
    x1= np.random.uniform(domain[0][0], domain[0][1], (1,1))

    x_plus= x_test+0.00001; x_minus= x_test-0.00001

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()

    L0_x1= np.linalg.cholesky(var0_x1); y1= u0_x1+ np.matmul(L0_x1,Z)

    du0_x1_dx1_test, dK0_x1_dx1_test, K0_x1_test= mean_variance_gradients(kernel, Xt, Yt, x_test, noise)
    du0_x1_dx1_plus, dK0_x1_dx1_plus, K0_x1_plus= mean_variance_gradients(kernel, Xt, Yt, x_plus, noise)
    du0_x1_dx1_minus, dK0_x1_dx1_minus, K0_x1_minus= mean_variance_gradients(kernel, Xt, Yt, x_minus, noise)

    dsigma0_dx1_test, sigma0_test, dL0_x1_dx1_test, L0_x1_test = \
        dsigma0_dx1_q1(x2, x_test, Xt, kernel, noise, K0_x1_test, dK0_x1_dx1_test, D, Yt)

    L0_x1_plus= np.linalg.cholesky(K0_x1_plus); L0_x1_minus= np.linalg.cholesky(K0_x1_minus)

    app_dL0_x1_dx1_test= ((L0_x1_plus- L0_x1_test)/(x_plus- x_test)+ \
                          (L0_x1_test- L0_x1_minus)/(x_test- x_minus))/2

    print('dL0_x1_dx1_test:{}, app_dL0_x1_dx1_test:{}'.format(dL0_x1_dx1_test, app_dL0_x1_dx1_test))

    K_x2_x_test, dK_x1_x_dx_test= posterior_covariance_and_derivative_1q(x2, x_test, Xt, kernel, noise)
    K_x2_x_plus, dK_x1_x_dx_plus= posterior_covariance_and_derivative_1q(x2, x_plus, Xt, kernel, noise)
    K_x2_x_minus, dK_x1_x_dx_minus= posterior_covariance_and_derivative_1q(x2, x_minus, Xt, kernel, noise)


    sigma0_x1_test= np.matmul(K_x2_x_test, np.linalg.inv(np.linalg.cholesky(K0_x1_test)))
    sigma0_x1_plus= np.matmul(K_x2_x_plus, np.linalg.inv(np.linalg.cholesky(K0_x1_plus)))
    sigma0_x1_minus= np.matmul(K_x2_x_minus, np.linalg.inv(np.linalg.cholesky(K0_x1_minus)))

    print('sigma0 test:{}, {}'.format(sigma0_test, sigma0_x1_test))
    app_grad= ((sigma0_x1_plus- sigma0_x1_test)/(x_plus- x_test)+ (sigma0_x1_test- sigma0_x1_minus)/(x_test- x_minus))/2

    print('dsimga0_dx1:{}\napp_grad:{}'.format(dsigma0_dx1_test, app_grad))


def lookahead_mean_and_variance_gradient_wrt_x1(dsigma0_dx1, sigma0, Z, D):

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv = Z.copy()

    du1_x2_dx1= np.matmul(dsigma0_dx1, Z_conv)

    dvar1_x2_dx1= np.empty([D,1])

    '''holds also for q>1'''
    for i in range(D):
        dsigma0_dx1i= dsigma0_dx1[i, :].reshape(-1,1)
        dvar1_x2_dx1i= -2* np.matmul(sigma0, np.transpose(dsigma0_dx1i))
        dvar1_x2_dx1[i, 0]= dvar1_x2_dx1i

    return du1_x2_dx1, dvar1_x2_dx1

def temp_test_lookahead_mean_and_variance_gradient(Z, x_test, model, domain, Xt, noise, kernel, D, Yt):
    x_plus= x_test+0.00001; x_minus= x_test-0.00001
    x2 = np.random.uniform(domain[0][0], domain[0][1], (1, 1))

    # u0_x_test, var0_x_test= model.predict_f(x_test); u0_x_test= u0_x_test.numpy(); var0_x_test= var0_x_test.numpy()
    # u0_x1_plus, var0_x1_plus= model.predict_f(x_plus); u0_x1_plus= u0_x1_plus.numpy(); var0_x1_plus= var0_x1_plus.numpy()
    # u0_x1_minus, var0_x1_minus= model.predict_f(x_plus); u0_x1_minus= u0_x1_minus.numpy(); var0_x1_minus= var0_x1_minus.numpy()

    u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

    if type(Z) != np.ndarray:
        Z = np.atleast_2d(Z)

    u1_x2_x_test, var1_x2_x_test= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_test, Z, Xt, noise)
    u1_x2_x_plus, var1_x2_x_plus= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_plus, Z, Xt, noise)
    u1_x2_x_minus, var1_x2_x_minus= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_minus, Z, Xt, noise)

    du0_x1_dx1_test, dK0_x1_dx1_test, K0_x1_test =  mean_variance_gradients(kernel, Xt, Yt, x_test, noise)

    dsigma0_dx1, sigma0, dL0_x1_dx1, L0_x1 = dsigma0_dx1_q1(x2, x_test, Xt, kernel, noise,
                                                            K0_x1_test, dK0_x1_dx1_test, D, Yt)

    du1_x2_x_test, dvar1_x2_x_test = lookahead_mean_and_variance_gradient_wrt_x1(dsigma0_dx1, sigma0, Z, D)

    # K0_x2_x_plus, _ = posterior_covariance_and_derivative_1q(x2, x_plus, Xt, kernel, noise)
    # K0_x2_x_minus, _ = posterior_covariance_and_derivative_1q(x2, x_minus, Xt, kernel, noise)

    app_du1_x2_x_test = ((u1_x2_x_plus - u1_x2_x_test)/(x_plus - x_test) + (u1_x2_x_test - u1_x2_x_minus)/(x_test - x_minus)) / 2
    app_dvar1_x2_x_test = ((var1_x2_x_plus - var1_x2_x_test) / (x_plus - x_test) + (var1_x2_x_test - var1_x2_x_minus)/(x_test - x_minus)) / 2
    # L0_x_test= np.linalg.cholesky(var_x_test); u1_x2_test= u0_x2 + np.matmul(L0_x2_test,Z); f_x1min= np.min(f_x1)
    #
    # u1_x2_x_test= u0_x2+ np.matmul(sigma0_x_test, Z)
    # u1_x2_x_plus= u0_x2+ np.matmul(sigma0_x_plus, Z)
    # u1_x2_x_minus= u0_x2+ np.matmul(sigma0_x_minus, Z)


    print('app_du1_x2_x_test:{},du1_x2_x_test:{}', app_du1_x2_x_test, du1_x2_x_test)
    print('app_dvar1_x2_x_test:{},dvar1_x2_x_test:{}', app_dvar1_x2_x_test, dvar1_x2_x_test)

def df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1):

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv = Z.copy()

    if (f_best- (u0_x1+ np.matmul(L0_x1,Z_conv))) <0:

        best1_grad= 0

    else:
        best1_grad= du0_x1_dx1+ np.matmul(dL0_x1_dx1, Z_conv)

    return best1_grad

def dEI1_x2_dx1(f_best, f_best1,  kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x1, u0_x2, var0_x2):

    # du_dx, dvar_dx, _= mean_variance_gradients(kernel, Xt, Yt, x, noise)
    du0_x1_dx1, dK0_x1_dx1, K0_x1 =  mean_variance_gradients(kernel, Xt, Yt, x1, noise)
    dsigma0_dx1, sigma0, dL0_x1_dx1, L0_x1= dsigma0_dx1_q1(x2_opt, x1, Xt, kernel, noise, K0_x1, dK0_x1_dx1, D, Yt)
    du1_x2_opt_dx1, dK1_x2_opt_dx1= lookahead_mean_and_variance_gradient_wrt_x1(dsigma0_dx1, sigma0, Z, D)

    best1_grad_x1= df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)

    u1_x2_opt, var1_x2_opt= one_step_mean_and_variance(x2_opt, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)
    sigma1_x2_opt= np.sqrt(var1_x2_opt)
    dsigma1_x2_opt_dx1 = (1 / (2 * sigma1_x2_opt)) * dK1_x2_opt_dx1

    '''cumulative gradient'''
    dcumulative_dx= dcumulative_du(f_best1, sigma1_x2_opt,u1_x2_opt)* du1_x2_opt_dx1 + \
                            dcumulative_dsigma(f_best1,sigma1_x2_opt,u1_x2_opt)*dsigma1_x2_opt_dx1+\
                                dcumulative_dfbest(f_best1,sigma1_x2_opt,u1_x2_opt)*best1_grad_x1

    # print('dcumulative_du:{}, du1_x2_dx:{}, dcumulative_dsigma:{}, dsigma1_x2_opt_dx1:{}, dcumulative_dfbest:{}'
    #       '*best1_grad_x1:{}'.format(dcumulative_du(f_best1, sigma1_x2_opt,u1_x2_opt), du1_x2_opt_dx1,
    #      dcumulative_dsigma(f_best1, sigma1_x2_opt, u1_x2_opt),dsigma1_x2_opt_dx1,
    #          dcumulative_dfbest(f_best1,sigma1_x2_opt,u1_x2_opt), best1_grad_x1))



    term1= (f_best1- u1_x2_opt)*dcumulative_dx

    term2= (best1_grad_x1-du1_x2_opt_dx1)* norm.cdf(((f_best1- u1_x2_opt)/sigma1_x2_opt))

    term3= dsigma1_x2_opt_dx1* norm.pdf((f_best1- u1_x2_opt)/sigma1_x2_opt)

    '''normal distribution gradient'''

    dnormal_dx= dnormal_du(f_best1, u1_x2_opt, sigma1_x2_opt)*du1_x2_opt_dx1+ \
                                dnormal_dsigma(f_best1, u1_x2_opt, sigma1_x2_opt)*dsigma1_x2_opt_dx1+\
                        dnormal_dfbest(f_best1, u1_x2_opt, sigma1_x2_opt)*best1_grad_x1

    term4= sigma1_x2_opt* dnormal_dx

    # print('term1:{}, term2:{}, term3:{}, term4:{}'.format(term1, term2, term3, term4))
    return term1+ term2+ term3+ term4

def temp_test_dEI1_x2_dx1_D1(x_test, f_best, kernel, Xt, Yt, noise, D, Z, domain, model):

    if type(Z) != np.ndarray:
        Z = np.atleast_2d(Z)

    x2 = np.random.uniform(domain[0][0], domain[0][1], (1, 1))

    '''analytical gradient EI'''
    u0_x2, var0_x2 = model.predict_f(x2); u0_x2 = u0_x2.numpy();  var0_x2 = var0_x2.numpy()

    u0_x1_test, var0_x1_test = model.predict_f(x_test); u0_x1_test = u0_x1_test.numpy(); var0_x1_test = var0_x1_test.numpy()

    f_best1_x_test= one_step_f_best(f_best, u0_x1_test, var0_x1_test, Z)

    dEI1_x2_dx1_test= dEI1_x2_dx1(f_best, f_best1_x_test, kernel, Xt, Yt, noise, x2, x_test, D, Z, u0_x1_test, u0_x2, var0_x2)


    '''approximate gradient EI'''
    x_plus = x_test + 0.0001; x_minus = x_test - 0.0001

    u0_x1_plus, var0_x1_plus = model.predict_f(x_plus);  u0_x1_plus = u0_x1_plus.numpy();
    var0_x1_plus = var0_x1_plus.numpy();
    u0_x1_minus, var0_x1_minus = model.predict_f(x_minus); u0_x1_minus = u0_x1_minus.numpy();
    var0_x1_minus = var0_x1_minus.numpy()

    f_best1_x_plus= one_step_f_best(f_best, u0_x1_plus, var0_x1_plus, Z)
    f_best1_x_minus= one_step_f_best(f_best, u0_x1_minus, var0_x1_minus, Z)

    u1_x2_x_test, var1_x2_x_test= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_test, Z, Xt, noise)
    sigma1_x2_x_test= np.sqrt(var1_x2_x_test)
    u1_x2_x_plus, var1_x2_x_plus= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_plus, Z, Xt, noise)
    sigma1_x2_x_plus= np.sqrt(var1_x2_x_plus)
    u1_x2_x_minus, var1_x2_x_minus= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_minus, Z, Xt, noise)
    sigma1_x2_x_minus= np.sqrt(var1_x2_x_minus)

    EI1_x2_x_test= EI(sigma1_x2_x_test, u1_x2_x_test, f_best1_x_test)
    EI1_x2_x_plus= EI(sigma1_x2_x_plus, u1_x2_x_plus, f_best1_x_plus)
    EI1_x2_x_minus= EI(sigma1_x2_x_minus, u1_x2_x_minus, f_best1_x_minus)

    app_deriv= ((EI1_x2_x_plus- EI1_x2_x_test)/(x_plus- x_test)+ (EI1_x2_x_test- EI1_x2_x_minus)/(x_test- x_minus))/2


    print('analytical grad:{}, app grad:{}'.format(dEI1_x2_dx1_test, app_deriv))


def temp_test_dcumulative_dx(Z, x_test, x2, model, domain, kernel, Xt, noise, Yt, D, f_best):

    x_plus= x_test+0.0001; x_minus= x_test-0.0001
    # x2 = np.random.uniform(domain[0][0], domain[0][1], (1, 1))

    u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

    '''approximate derrivatives'''
    u0_x1_test, var0_x1_test= model.predict_f(x_test); u0_x1_test= u0_x1_test.numpy(); var0_x1_test= var0_x1_test.numpy()
    u0_x1_plus, var0_x1_plus= model.predict_f(x_plus); u0_x1_plus= u0_x1_plus.numpy(); var0_x1_plus= var0_x1_plus.numpy()
    u0_x1_minus, var0_x1_minus= model.predict_f(x_minus); u0_x1_minus= u0_x1_minus.numpy(); var0_x1_minus= var0_x1_minus.numpy()

    if type(Z) != np.ndarray:
        Z = np.atleast_2d(Z)

    u1_x2_x_test, var1_x2_x_test= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_test, Z, Xt, noise)
    u1_x2_x_plus, var1_x2_x_plus= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_plus, Z, Xt, noise)
    u1_x2_x_minus, var1_x2_x_minus= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x_minus, Z, Xt, noise)

    sigma1_x2_x_test= np.sqrt(var1_x2_x_test); sigma1_x2_x_plus= np.sqrt(var1_x2_x_plus)
    sigma1_x2_x_minus= np.sqrt(var1_x2_x_minus)

    f_best1_x_test =one_step_f_best(f_best, u0_x1_test, var0_x1_test, Z)
    f_best1_x_plus =one_step_f_best(f_best, u0_x1_plus, var0_x1_plus, Z)
    f_best1_x_minus =one_step_f_best(f_best, u0_x1_minus, var0_x1_minus, Z)

    cum_x_test= norm.cdf((f_best1_x_test- u1_x2_x_test)/sigma1_x2_x_test)
    cum_x_plus= norm.cdf((f_best1_x_plus- u1_x2_x_plus)/sigma1_x2_x_plus)
    cum_x_minus= norm.cdf((f_best1_x_minus- u1_x2_x_minus)/sigma1_x2_x_minus)

    app_deriv= ((cum_x_plus- cum_x_test)/(x_plus- x_test)+ (cum_x_test- cum_x_minus)/(x_test- x_minus))/2

    '''calculate analytical derrivative'''
    du0_x1_dx_test, dK0_x1_dx_test, K0_x_test = mean_variance_gradients(kernel, Xt, Yt, x_test, noise)
    du0_x1_dx_plus, dK0_x1_dx_plus, K0_x_plus = mean_variance_gradients(kernel, Xt, Yt, x_plus, noise)
    du0_x1_dx_minus, dK0_x1_dx_minus, K0_x_minus = mean_variance_gradients(kernel, Xt, Yt, x_minus, noise)

    dsigma0_dx_test, sigma0, dL0_x1_dx_test, L0_x_test= dsigma0_dx1_q1(x2, x_test, Xt, kernel, noise, var0_x1_test,
                                                                       dK0_x1_dx_test, D, Yt)
    dsigma0_dx_plus, sigma0_plus, dL0_x1_dx_plus, L0_x_plus= dsigma0_dx1_q1(x2, x_plus, Xt, kernel, noise, var0_x1_plus,
                                                                       dK0_x1_dx_plus, D, Yt)
    dsigma0_dx_minus, sigma0_minus, dL0_x1_dx_minus, L0_x_minus= dsigma0_dx1_q1(x2, x_minus, Xt, kernel, noise, var0_x1_minus,
                                                                       dK0_x1_dx_minus, D, Yt)

    '''calculate du_dx and dvar_dx'''
    du1_x2_dx_test, dK1_x2_dx_test= lookahead_mean_and_variance_gradient_wrt_x1(dsigma0_dx_test, sigma0, Z, D)
    dsigma1_x2_dx_test = (1 / (2 * sigma1_x2_x_test)) * dK1_x2_dx_test


    '''calculate dfbest1_dx'''
    best1_grad_x1_test = df_best1_dx1(f_best1_x_test, x_test, u0_x1_test,L0_x_test, Z, du0_x1_dx_test,  dL0_x1_dx_test)

    #test du0_dx1 and dL0_dx1
    app_deriv_u0_x1= ((u0_x1_plus- u0_x1_test)/(x_plus- x_test)+ (u0_x1_test - u0_x1_minus)/(x_test- x_minus))/2
    print('app_deriv_u0_x1:{}, analytical_deriv_u0_x1:{}'.format(app_deriv_u0_x1, du0_x1_dx_test))
    app_deriv_L0_x1= ((L0_x_plus- L0_x_test)/(x_plus- x_test)+ (L0_x_test - L0_x_minus)/(x_test- x_minus))/2
    print('app_deriv_L0_x1:{}, analytical_deriv_L0_x1:{}'.format(app_deriv_L0_x1, dL0_x1_dx_test))

    print('u0_x1_minus:{}, u0_x1:{}, u0_x1_plus:{}'.format(u0_x1_minus, u0_x1_test, u0_x1_plus))
    print('L0_x1_minus:{}, L0_x1:{}, L0_x1_plus:{}'.format(L0_x_minus, L0_x_test, L0_x_plus))
    print('f_best1_x_minus:{}, f_best1_x_test:{}, f_best1_x_plus:{}'.format(f_best1_x_minus, f_best1_x_test, f_best1_x_plus))

    '''analytical derrivative'''
    dcumulative_dx= dcumulative_du(f_best1_x_test, sigma1_x2_x_test, u1_x2_x_test)*du1_x2_dx_test + \
                            dcumulative_dsigma(f_best1_x_test,sigma1_x2_x_test,u1_x2_x_test)*dsigma1_x2_dx_test+\
                                dcumulative_dfbest(f_best1_x_test,sigma1_x2_x_test,u1_x2_x_test)*best1_grad_x1_test

    '''compare derivatives'''
    print('app_deriv:{},dcumulative_dx:{}', app_deriv, dcumulative_dx)

    '''analytical gradient term1+term2'''

    dcumulative_dx = dcumulative_du(f_best1_x_test, sigma1_x2_x_test, u1_x2_x_test) * du1_x2_dx_test + \
                     dcumulative_dsigma(f_best1_x_test, sigma1_x2_x_test, u1_x2_x_test) * dsigma1_x2_dx_test + \
                     dcumulative_dfbest(f_best1_x_test, sigma1_x2_x_test, u1_x2_x_test) * best1_grad_x1_test

    term1 = (f_best1_x_test - u1_x2_x_test) * dcumulative_dx
    term2 = (best1_grad_x1_test - du1_x2_dx_test) * norm.cdf(((f_best1_x_test - u1_x2_x_test) / sigma1_x2_x_test))

    d_term12_dx= term1+term2

    '''term2 comparison of analytical and approximate derivatives'''
    #f_best1_deriv
    app_deriv_f_best1= ((f_best1_x_plus- f_best1_x_test)/(x_plus- x_test)+ (f_best1_x_test - f_best1_x_minus)/(x_test- x_minus))/2
    print('app_f_best1_deriv:{}, analytical_f_best1_deriv:{}'.format(app_deriv_f_best1, best1_grad_x1_test))
    app_deriv_f_best1_right= (f_best1_x_plus- f_best1_x_test)/(x_plus- x_test)
    app_deriv_f_best1_left= (f_best1_x_test - f_best1_x_minus)/(x_test- x_minus)
    print('app_f_best1_deriv_right:{}, app_f_best1_deriv_left:{}, analytical_f_best1_deriv:{}'
          .format(app_deriv_f_best1_right, app_deriv_f_best1_left, best1_grad_x1_test))


    #u1_x2 deriv
    app_deriv_u1_x2= ((u1_x2_x_plus- u1_x2_x_test)/(x_plus- x_test)+(u1_x2_x_test- u1_x2_x_minus)/(x_test- x_minus))/2

    print('app_u1_x2_deriv:{}, analytical_u1_x2_deriv:{}'.format(app_deriv_u1_x2, du1_x2_dx_test))

    app_deriv_right = (cum_x_plus - cum_x_test) / (x_plus - x_test)
    app_deriv_left= (cum_x_test - cum_x_minus) / (x_test - x_minus)
    print('app_cum_deriv_left{}, app_cum_deriv_right:{}, analytical_cum_deriv:{}'.format(app_deriv_left,
                                                             app_deriv_right, dcumulative_dx))


    '''approximate gradient term1+term2'''
    term1_x_test= (f_best1_x_test-u1_x2_x_test)*norm.cdf(((f_best1_x_test - u1_x2_x_test) / sigma1_x2_x_test))
    term1_x_plus= (f_best1_x_plus-u1_x2_x_plus)*norm.cdf(((f_best1_x_plus - u1_x2_x_plus) / sigma1_x2_x_plus))
    term1_x_minus= (f_best1_x_minus-u1_x2_x_minus)*norm.cdf(((f_best1_x_minus - u1_x2_x_minus) / sigma1_x2_x_minus))

    app_deriv_term12= ((term1_x_plus- term1_x_test)/(x_plus- x_test)+ (term1_x_test- term1_x_minus)/(x_test- x_minus))/2
    print('app_term12_deriv:{}, analytical_term12_deriv:{}'.format(app_deriv_term12, d_term12_dx))



    '''analytical gradient term3+term4'''

    term3= dsigma1_x2_dx_test* norm.pdf((f_best1_x_test- u1_x2_x_test)/sigma1_x2_x_test)

    '''normal distribution gradient'''

    dnormal_dx= dnormal_du(f_best1_x_test, u1_x2_x_test, sigma1_x2_x_test)*du1_x2_dx_test+ \
                                dnormal_dsigma(f_best1_x_test, u1_x2_x_test, sigma1_x2_x_test)*dsigma1_x2_dx_test+\
                        dnormal_dfbest(f_best1_x_test, u1_x2_x_test, sigma1_x2_x_test)*best1_grad_x1_test

    term4= sigma1_x2_x_test* dnormal_dx

    d_term34_dx= term3+term4

    '''approximate gradient term3+term4'''
    term34_x_test= (sigma1_x2_x_test)*norm.pdf(((f_best1_x_test - u1_x2_x_test) / sigma1_x2_x_test))
    term34_x_plus= (sigma1_x2_x_plus)*norm.pdf(((f_best1_x_plus - u1_x2_x_plus) / sigma1_x2_x_plus))
    term34_x_minus= (sigma1_x2_x_minus)*norm.pdf(((f_best1_x_minus - u1_x2_x_minus) / sigma1_x2_x_minus))

    app_deriv_term34= ((term34_x_plus- term34_x_test)/(x_plus- x_test)+ (term34_x_test- term34_x_minus)/(x_test- x_minus))/2

    print('app_term34_deriv:{}, analytical_term34_deriv:{}'.format(app_deriv_term34, d_term34_dx))

    '''approximate gradient using another functioin'''
    EI1_x2_x_test = EI(sigma1_x2_x_test, u1_x2_x_test, f_best1_x_test)
    EI1_x2_x_plus = EI(sigma1_x2_x_plus, u1_x2_x_plus, f_best1_x_plus)
    EI1_x2_x_minus = EI(sigma1_x2_x_minus, u1_x2_x_minus, f_best1_x_minus)

    app_deriv_EI_other= ((EI1_x2_x_plus- EI1_x2_x_test)/(x_plus- x_test)+ (EI1_x2_x_test- EI1_x2_x_minus)/(x_test- x_minus))/2

    app_deriv_EI= app_deriv_term12+ app_deriv_term34;

    analytical_deriv= d_term12_dx+ d_term34_dx
    print('term1:{}, term:{}, term3:{}, term4:{}'.format(term1, term2, term3, term4))
    analytical_deriv_other= dEI1_x2_dx1(f_best, f_best1_x_test, kernel, Xt, Yt, noise, x2, x_test, D, Z, u0_x1_test, u0_x2, var0_x2)
    print('app_deriv_EI:{}, app_deriv_EI_other:{}, analytical_deriv:{}, analytical_deriv_other:{}'.format(app_deriv_EI,
                                                                   app_deriv_EI_other, analytical_deriv, analytical_deriv_other))


def two_opt_EI_optimize(f_best, Xt, Yt, model, domain, noise, kernel, num_inner_opt_restarts,
                                                                num_outer_opt_restarts, monte_carlo_samples,  D, Q=1):


    def two_step_negative_improvement(x1, Z):

        x1 = x1.reshape(1,-1)
        print('x1 shape:{}'.format(x1.shape))
        '''observation at x1'''
        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        u0_x1, var0_x1 = model.predict_f(x1);
        u0_x1 = u0_x1.numpy();
        var0_x1 = var0_x1.numpy()
        sigma0_x1 = np.sqrt(var0_x1)

        L0_x1=  np.linalg.cholesky(var0_x1)
        y1= u0_x1+ np.matmul(L0_x1,Z_conv)
        '''update Xt and Yt according to outcome of Z'''
        Xt1= np.append(Xt, x1, axis=0)
        Yt1= np.append(Yt, y1, axis=0)

        '''first two step improvement term'''
        i0_x1= np.maximum(f_best- float(np.min(Yt1, axis=0)), 0)


        '''inner loop optimization with random restarts'''
        x2_opt_list = [];
        opt_value_list = [];
        grad_list = [];
        x20_list = []
        lower = [domain[i][0] for i in range(len(domain))];
        upper = [domain[i][1] for i in range(len(domain))]

        for i in range(num_inner_opt_restarts):
            print('iter_inner:{}'.format(i))
            x20 = np.random.uniform(lower, upper, (1, D))
            x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt,
                                                                        noise, model, f_best, domain)

            x2_cand = x2_cand.reshape(1, -1)
            x2_opt_list.append(x2_cand);
            opt_value_list.append(cand_value);
            grad_list.append(x2_cand_grad)
            x20_list.append(x20)

        index_opt = int(np.argmax(np.array(opt_value_list)))

        # global x2_opt

        x2_opt = x2_opt_list[index_opt]
        x2_opt = x2_opt.reshape(1, -1)
        opt_value = opt_value_list[index_opt]
        x2_opt_grad = grad_list[index_opt]
        x20_opt = x20_list[index_opt]


        '''second two step improvement term'''
        ei1_x2= opt_value
        ei1_x2= float(ei1_x2)

        two_opt= i0_x1+ ei1_x2

        if type(two_opt)!= float:

            two_opt= two_opt.flatten()
        print('two_opt:{}'.format(two_opt))
        return -two_opt, x2_opt

    def two_step_negative_gradient(x1, Z, x2_opt):

        x1 = x1.reshape(1,-1)

        '''find cholesky gradient (L0_x1) and du0_x1_dx1'''
        du0_x1_dx1, dvar0_x1_dx1, K0_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

        u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
        sigma0_x1= np.sqrt(var0_x1)

        L0_x1= np.linalg.cholesky(var0_x1)

        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        if f_best- (u0_x1+ np.matmul(L0_x1, Z_conv))<0:

            term_grad1= 0
        else:

            # _, dK0_x1_dx1, K0_x1_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

            L0_x1 = np.linalg.cholesky(var0_x1); L0_x1_inv = np.linalg.inv(L0_x1);
            L0_x1_inv_T = np.transpose(L0_x1_inv)

            dL0_x1_dx1= np.empty([D,1])
            for i in range(D):
                dvar0_x1_dx1i= dvar0_x1_dx1[i,:].reshape(1,1)
                Ai= np.matmul(L0_x1_inv, np.matmul(dvar0_x1_dx1i, L0_x1_inv_T))
                temp= funfi(Ai)
                dL0_x1_dx1i = np.matmul(L0_x1, temp)
                dL0_x1_dx1[i,0]= dL0_x1_dx1i

            # term_grad1= -df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)
            term_grad1= -du0_x1_dx1- np.matmul(dL0_x1_dx1, Z_conv)

        # global x2_opt

        '''gradient of lookahead term'''
        u0_x2_opt, var0_x2_opt= model.predict_f(x2_opt); u0_x2_opt= u0_x2_opt.numpy(); var0_x2_opt= var0_x2_opt.numpy()
        sigma0_x1= np.sqrt(var0_x1)


        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        y1= u0_x1+ np.matmul(L0_x1, Z_conv)

        f_best1= np.minimum(f_best, float(y1))

        EI1_x2_grad_x1= dEI1_x2_dx1(f_best, f_best1, kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x1, u0_x2_opt,
                    var0_x2_opt)

        print('term_grad1:{}, EI1_x2_grad_x1:{}'.format(term_grad1, EI1_x2_grad_x1))
        two_step_grad= term_grad1+ EI1_x2_grad_x1
        two_step_grad= two_step_grad.flatten()
        print('two_step_grad:{}'.format(two_step_grad))
        return -two_step_grad

    Z_list = np.random.normal(0.0, 1.0, (Q, monte_carlo_samples))


    def monte_carlo_average_two_step_improvement(x1):

        avg_negative_improvement= 0

        global x2_opt_list
        x2_opt_list= np.zeros([Z_list.shape[1], D])

        for k in range(Z_list.shape[1]):

            Z= Z_list[:, k]
            negative_improvementk, x2_opt = two_step_negative_improvement(x1,Z)
            avg_negative_improvement+= negative_improvementk/monte_carlo_samples

            x2_opt_list[k, :]= x2_opt[0, :]


        return avg_negative_improvement

    def monte_carlo_average_two_step_gradient(x1):

        avg_negative_gradient = 0

        global x2_opt_list

        for k in range(Z_list.shape[1]):

            x2_opt= x2_opt_list[k, :].reshape(1,-1)

            Z = Z_list[:, k]
            negative_improvementk = two_step_negative_gradient(x1,Z, x2_opt)
            avg_negative_gradient += negative_improvementk / monte_carlo_samples

        return avg_negative_gradient

    lower= []; upper= []
    for i in range(len(domain)):
        lower.append(domain[i][0])
        upper.append(domain[i][1])
    b= Bounds(lb= lower, ub= upper )

    x1_opt_list = [];
    opt_value_list_x1 = [];
    grad_list_x1 = [];
    x10_list = []
    lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]


    for i in range(num_outer_opt_restarts):

        print('iter_outer:{}'.format(i))
        x10= np.random.uniform(lower, upper, (1,D))
        print('x10 shape:{}'.format(x10.shape))
        result= (minimize(monte_carlo_average_two_step_improvement, x10, bounds=b,  method='L-BFGS-B',
                          jac=monte_carlo_average_two_step_gradient))
        x1_cand= result['x']; cand_value_x1= result['fun']; x1_cand_grad= result['jac']
        cand_value_x1= -cand_value_x1; x1_cand_grad= - x1_cand_grad
        x1_cand = x1_cand.reshape(1, -1)

        x1_opt_list.append(x1_cand);
        opt_value_list_x1.append(cand_value_x1);
        grad_list_x1.append(x1_cand_grad)
        x10_list.append(x10)

    index_opt_x1 = int(np.argmax(np.array(opt_value_list_x1)))
    x1_opt = x1_opt_list[index_opt_x1]
    x1_opt= x1_opt.reshape(1,-1)
    opt_value_x1 = opt_value_list_x1[index_opt_x1]
    x1_opt_grad = grad_list_x1[index_opt_x1]
    x10_opt = x10_list[index_opt_x1]


    return x1_opt, opt_value_x1,  x1_opt_grad, result



def get_two_step_improvement_only(x1,f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D):


    '''observation at x1'''
    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    u0_x1, var0_x1 = model.predict_f(x1); u0_x1 = u0_x1.numpy();
    var0_x1 = var0_x1.numpy()
    sigma0_x1 = np.sqrt(var0_x1)

    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)

    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    '''first two step improvement term'''
    i0_x1= np.maximum(f_best- float(np.min(Yt1, axis=0)), 0)


    '''inner loop optimization with random restarts'''
    x2_opt_list = [];
    opt_value_list = [];
    grad_list = [];
    x20_list = []
    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    for i in range(num_inner_opt_restarts):

        x20 = np.random.uniform(lower, upper, (1, D))
        x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt,
                                                                    noise, model, f_best, domain)

        x2_cand = x2_cand.reshape(1, -1)
        x2_opt_list.append(x2_cand);
        opt_value_list.append(cand_value);
        grad_list.append(x2_cand_grad)
        x20_list.append(x20)

    index_opt = int(np.argmax(np.array(opt_value_list)))

    x2_opt_loc = x2_opt_list[index_opt]
    opt_value = opt_value_list[index_opt]
    x2_opt_grad = grad_list[index_opt]
    x20_opt = x20_list[index_opt]

    print('x2_opt:{}'.format(x2_opt))
    '''second two step improvement term'''
    ei1_x2= float(opt_value)

    two_opt= i0_x1+ ei1_x2


    return two_opt


def inner_loop_optimization(num_inner_opt_restarts, domain, model, kernel, noise, f_best, u0_x1, var0_x1, x1,
                                        Z, Xt, Yt, D):

    '''inner loop optimization with random restarts'''
    x2_opt_list = [];
    opt_value_list = [];
    grad_list = [];
    x20_list = []
    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    for i in range(num_inner_opt_restarts):

        x20 = np.random.uniform(lower, upper, (1, D))
        x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt,
                                                                    noise, model, f_best, domain)

        x2_cand = x2_cand.reshape(1, -1)
        x2_opt_list.append(x2_cand);
        opt_value_list.append(cand_value);
        grad_list.append(x2_cand_grad)
        x20_list.append(x20)

    index_opt = int(np.argmax(np.array(opt_value_list)))

    x2_opt_loc = x2_opt_list[index_opt]
    opt_value = opt_value_list[index_opt]
    x2_opt_grad = grad_list[index_opt]
    x20_opt = x20_list[index_opt]

    return x2_opt, opt_value, x2_opt_grad

def get_EI1_x2_only(x2, x1, Z, model, noise, kernel, var0_x1, u0_x1, Xt, Yt, f_best):

    '''observation at x1'''

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    sigma0_x1= np.sqrt(var0_x1)
    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)
    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    '''update f_best according to the outcome of Z'''

    f_best1 = one_step_f_best(f_best, u0_x1, var0_x1, Z_conv)

    # print('x2 shape in obj', x2.shape)
    u0_x2, var0_x2= model.predict_f(x2); u0_x2= u0_x2.numpy(); var0_x2= var0_x2.numpy()

    u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)

    sigma1_x2= np.sqrt(var1_x2)

    EI1_x2= EI(sigma1_x2, u1_x2, f_best1)

    return EI1_x2

def temp_test_EI1x2(x1, x2, model, Z, kernel, noise, Xt, Yt, f_best):

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    EI1_x2= get_EI1_x2_only(x2, x1, Z, model, noise, kernel, var0_x1, u0_x1, Xt, Yt, f_best)

    return EI1_x2

def temp_test_dEI1_x2_dx1(x_test, x2, domain, f_best, Xt, Yt, kernel, noise, model, D, Z):

    # x_test= np.random.uniform(domain[0][0], domain[0][1], (1,1));
    # x2= np.random.uniform(domain[0][0], domain[0][1], (1,1))

    x_plus = x_test + 0.0001;
    x_minus = x_test - 0.0001

    EI1_x2_x_test= temp_test_EI1x2(x_test, x2, model, Z, kernel, noise, Xt, Yt, f_best)
    EI1_x2_x_plus= temp_test_EI1x2(x_plus, x2, model, Z, kernel, noise, Xt, Yt, f_best)
    EI1_x2_x_minus= temp_test_EI1x2(x_minus, x2, model, Z, kernel, noise, Xt, Yt, f_best)

    u0_x_test, var0_x_test = model.predict_f(x_test); u0_x_test = u0_x_test.numpy();  var0_x_test = var0_x_test.numpy()
    u0_x2, var0_x2 = model.predict_f(x2); u0_x2 = u0_x2.numpy();  var0_x2 = var0_x2.numpy()

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    f_best1_x_test = one_step_f_best(f_best, u0_x_test, var0_x_test, Z_conv)
    test_grad= dEI1_x2_dx1(f_best, f_best1_x_test,  kernel, Xt, Yt, noise, x2, x_test, D, Z, u0_x_test, u0_x2, var0_x2)


    app_grad= ((EI1_x2_x_plus- EI1_x2_x_test)/(x_plus- x_test)+ (EI1_x2_x_test- EI1_x2_x_minus)/(x_test- x_minus))/2

    print('test_grad:{}\napp_grad:{}'.format(test_grad, app_grad))

def get_two_step_gradient_only(x1, kernel, Xt, Yt, noise, model, Z, f_best, num_inner_opt_restarts, domain, D):

    '''find cholesky gradient (L0_x1) and du0_x1_dx1'''
    du0_x1_dx1, dvar0_x1_dx1, K0_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    sigma0_x1= np.sqrt(var0_x1)

    L0_x1= np.linalg.cholesky(var0_x1)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    if f_best- (u0_x1+ np.matmul(L0_x1, Z_conv))<0:

        term_grad1= 0
    else:

        # _, dK0_x1_dx1, K0_x1_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

        L0_x1 = np.linalg.cholesky(var0_x1); L0_x1_inv = np.linalg.inv(L0_x1);
        L0_x1_inv_T = np.transpose(L0_x1_inv)

        dL0_x1_dx1 = np.empty([D, 1])
        for i in range(D):
            dvar0_x1_dx1i = dvar0_x1_dx1[i, :].reshape(1, 1)
            Ai = np.matmul(L0_x1_inv, np.matmul(dvar0_x1_dx1i, L0_x1_inv_T))
            temp = funfi(Ai)
            dL0_x1_dx1i = np.matmul(L0_x1, temp)
            dL0_x1_dx1[i, 0] = dL0_x1_dx1i

        # term_grad1= df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)
        term_grad1= du0_x1_dx1+ np.matmul(dL0_x1_dx1, Z_conv)


    x2_opt, opt_value, x2_opt_grad= inner_loop_optimization(num_inner_opt_restarts, domain, model, kernel,
                                        noise, f_best, u0_x1, var0_x1, x1,  Z, Xt, Yt, D)

    print('x2_opt:{}'.format(x2_opt))

    '''gradient of lookahead term'''
    u0_x2_opt, var0_x2_opt= model.predict_f(x2_opt); u0_x2_opt= u0_x2_opt.numpy(); var0_x2_opt= var0_x2_opt.numpy()
    sigma0_x1= np.sqrt(var0_x1)


    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    y1= u0_x1+ np.matmul(L0_x1, Z_conv)

    f_best1= np.minimum(f_best, float(y1))

    EI1_x2_grad_x1= dEI1_x2_dx1(f_best, f_best1, kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x1, u0_x2_opt,
                var0_x2_opt)

    two_step_grad= term_grad1+ EI1_x2_grad_x1

    return two_step_grad


def temp_test_two_step_improvement_only(x1,f_best, Xt, Yt, model, domain, Z, noise, kernel, x2, D):


    '''observation at x1'''
    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    u0_x1, var0_x1 = model.predict_f(x1); u0_x1 = u0_x1.numpy();
    var0_x1 = var0_x1.numpy()
    sigma0_x1 = np.sqrt(var0_x1)

    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)

    '''update Xt and Yt according to outcome of Z'''
    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    '''first two step improvement term'''
    i0_x1= np.maximum(f_best- float(np.min(Yt1, axis=0)), 0)


    # '''inner loop optimization with random restarts'''
    # x2_opt_list = [];
    # opt_value_list = [];
    # grad_list = [];
    # x20_list = []
    # lower = [domain[i][0] for i in range(len(domain))];
    # upper = [domain[i][1] for i in range(len(domain))]

    # for i in range(num_inner_opt_restarts):
    #
    #     x20 = np.random.uniform(lower, upper, (1, D))
    #     x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt,
    #                                                                 noise, model, f_best, domain)
    #
    #     x2_cand = x2_cand.reshape(1, -1)
    #     x2_opt_list.append(x2_cand);
    #     opt_value_list.append(cand_value);
    #     grad_list.append(x2_cand_grad)
    #     x20_list.append(x20)
    #
    # index_opt = int(np.argmax(np.array(opt_value_list)))
    #
    # x2_opt_loc = x2_opt_list[index_opt]
    # opt_value = opt_value_list[index_opt]
    # x2_opt_grad = grad_list[index_opt]
    # x20_opt = x20_list[index_opt]


    '''second two step improvement term'''
    u0_x2, var0_x2 = model.predict_f(x2); u0_x2 = u0_x2.numpy(); var0_x2 = var0_x2.numpy()
    sigma0_x2 = np.sqrt(var0_x2)

    u1_x2, var1_x2= one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)
    sigma1_x2= np.sqrt(var1_x2)
    f_best1= np.minimum(f_best, y1)

    ei1_x2= float(EI(sigma1_x2, u1_x2, f_best1))
    ei1_x2_other= get_EI1_x2_only(x2, x1, Z, model, noise, kernel, var0_x1, u0_x1, Xt, Yt, f_best)

    print('ei1_x2:{}, ei1_x2_other:{}'.format(ei1_x2, ei1_x2_other))
    two_opt= i0_x1+ ei1_x2


    return two_opt, i0_x1, ei1_x2

def temp_test_two_step_gradient_only(x1, kernel, Xt, Yt, noise, model, Z, f_best, x2, domain, D):

    '''find cholesky gradient (L0_x1) and du0_x1_dx1'''
    du0_x1_dx1, dvar0_x1_dx1, K0_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    sigma0_x1= np.sqrt(var0_x1)

    L0_x1= np.linalg.cholesky(var0_x1)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    if f_best- (u0_x1+ np.matmul(L0_x1, Z_conv))<0:

        term_grad1= 0
    else:

        # _, dK0_x1_dx1, K0_x1_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

        L0_x1 = np.linalg.cholesky(var0_x1); L0_x1_inv = np.linalg.inv(L0_x1);
        L0_x1_inv_T = np.transpose(L0_x1_inv)

        dL0_x1_dx1 = np.empty([D, 1])
        for i in range(D):
            dvar0_x1_dx1i = dvar0_x1_dx1[i, :].reshape(1, 1)
            Ai = np.matmul(L0_x1_inv, np.matmul(dvar0_x1_dx1i, L0_x1_inv_T))
            temp = funfi(Ai)
            dL0_x1_dx1i = np.matmul(L0_x1, temp)
            dL0_x1_dx1[i, 0] = dL0_x1_dx1i

        # term_grad1= df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)
        term_grad1= -du0_x1_dx1- np.matmul(dL0_x1_dx1, Z_conv)


    # x2_opt, opt_value, x2_opt_grad= inner_loop_optimization(num_inner_opt_restarts, domain, model, kernel,
    #                                     noise, f_best, u0_x1, var0_x1, x1,  Z, Xt, Yt, D)

    x2_opt= x2.copy()

    '''gradient of lookahead term'''
    u0_x2_opt, var0_x2_opt= model.predict_f(x2_opt); u0_x2_opt= u0_x2_opt.numpy(); var0_x2_opt= var0_x2_opt.numpy()
    sigma0_x2_opt= np.sqrt(var0_x2_opt)


    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    y1= u0_x1+ np.matmul(L0_x1, Z_conv)

    f_best1= np.minimum(f_best, float(y1))

    EI1_x2_grad_x1= dEI1_x2_dx1(f_best, f_best1, kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x1, u0_x2_opt,
                var0_x2_opt)

    print('\nterm1_grad', term_grad1)
    print('EI1_x2_grad_x1', EI1_x2_grad_x1)
    two_step_grad= term_grad1+ EI1_x2_grad_x1

    '''approximation'''
    x1_pl= x1+0.00001; x1_mn= x1- 0.00001;
    x1_val, x1_i0_x1, x1_ei1_x2 = temp_test_two_step_improvement_only(x1,f_best, Xt, Yt, model, domain, Z, noise, kernel, x2, D)
    x_pl_val, x_pl_i0_x1, x_pl_ei1_x2 = temp_test_two_step_improvement_only(x1_pl,f_best, Xt, Yt, model, domain, Z, noise, kernel, x2, D)
    x_mn_val, x_mn_i0_x1, x_mn_ei1_x2= temp_test_two_step_improvement_only(x1_mn,f_best, Xt, Yt, model, domain, Z, noise, kernel, x2, D)

    app_deriv_term1=  ((x_pl_i0_x1- x1_i0_x1)/(x1_pl- x1)+\
                                    (x1_i0_x1- x_mn_i0_x1)/(x1- x1_mn))/2
    app_deriv_term2=  ((x_pl_ei1_x2- x1_ei1_x2 )/(x1_pl- x1)+\
                                    (x1_ei1_x2 - x_mn_ei1_x2)/(x1- x1_mn))/2

    print('\nterm_grad1:{}, app_term_grad1:{}'.format(term_grad1, app_deriv_term1))
    print('\nEI1_x2_grad_x1:{}, app_term_grad2:{}'.format(EI1_x2_grad_x1, app_deriv_term2))

    app_deriv= ((x_pl_val- x1_val )/(x1_pl- x1)+\
                                    (x1_val- x_mn_val)/(x1- x1_mn))/2

    print('\napp_deriv:\n{}\nopt_deriv:\n{}'.format(app_deriv , two_step_grad))
    return two_step_grad

