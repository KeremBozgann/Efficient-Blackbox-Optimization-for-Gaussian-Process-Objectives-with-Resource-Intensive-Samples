
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import gpflow as gp

def dcumulative_du(f_best, sigma_x, u_x):

    return -1/sigma_x*norm.pdf((f_best- u_x)/sigma_x)


def dcumulative_dsigma(f_best, sigma_x, u_x):

    return -(f_best-u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)


def dnormal_du(f_best, u_x, sigma_x):

    return (f_best- u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)


def dnormal_dsigma(f_best, u_x, sigma_x):

    return ((f_best-u_x)**2)/(sigma_x**3)*norm.pdf((f_best- u_x)/sigma_x)


def ei_pu(sigma_x, u_x, f_best, u_cost):
    gama_x = (f_best - u_x) / sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x = sigma_x * (gama_x * fi_x + norm.pdf(gama_x))

    EI_pu = EI_x / u_cost

    return EI_pu


def ei(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))

    return EI_x


def get_mean_variance_gradients_q1(kernel, Xt, Yt, x, noise):

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


def get_posterior_covariance(x, x1, Xt, kernel, noise):

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    temp= np.matmul((kernel.K(x,Xt)).numpy(), K_inv)
    result= (kernel.K(x,x1)).numpy()- np.matmul(temp, ((kernel.K(x1, Xt)).numpy()).T)

    return result

def get_one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise, var0_x1):


    k0_x2_x1= get_posterior_covariance(x2, x1, Xt, kernel,  noise)
    # k0_x1_x1= get_posterior_covariance(x1, x1, Xt, kernel, noise)
    L0_x1= np.linalg.cholesky(var0_x1)
    sigma0= np.matmul(k0_x2_x1, np.linalg.inv(L0_x1))

    u1_x2= u0_x2 + np.matmul(sigma0,Z)

    var1_x2= var0_x2- np.matmul(sigma0, sigma0.T)


    return u1_x2, var1_x2


def get_mean_var_std(x2, model):

    u0_x2, var0_x2 = model.predict_f(x2);
    u0_x2 = u0_x2.numpy();
    var0_x2 = var0_x2.numpy()
    sigma0_x2= np.sqrt(var0_x2)

    return u0_x2, var0_x2, sigma0_x2

def get_mean_of_cost(x2, latent_cost_model):

    u0_x2_latent_cost, var0_x2_latent_cost = latent_cost_model.predict_f(x2);
    u0_x2_latent_cost = u0_x2_latent_cost.numpy();
    u0_x2_cost = np.exp(u0_x2_latent_cost)

    return u0_x2_cost



def get_dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise):

    du_dx, dvar_dx, _= get_mean_variance_gradients_q1(kernel, Xt, Yt, x, noise)

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


def ei1_x2_pu_optimize_grid(X2_grid, kernel, noise, latent_cost_model, Xt1, Yt1):


    model1= gp.models.GPR((Xt1, Yt1), kernel)
    model1.likelihood.variance.assign(noise)


    '''update f_best according to the outcome of Z'''

    f_best1 = float(np.min(Yt1, axis=0))

    u1_X2_grid, var1_X2_grid, sigma1_X2_grid= get_mean_var_std(X2_grid, model1)

    u0_cost_X2_grid= get_mean_of_cost(X2_grid, latent_cost_model)

    EI1_X2_grid= ei(sigma1_X2_grid, u1_X2_grid, f_best1)

    EI1_pu_X2_grid= EI1_X2_grid/u0_cost_X2_grid

    index_max_grid= int(np.argmax(EI1_pu_X2_grid, axis=0))
    x2_opt= X2_grid[index_max_grid, :].reshape(1,-1)
    x2_opt_value= EI1_pu_X2_grid[index_max_grid, :].reshape(1,-1)

    '''EI1_x2_opt'''
    ei1_x2_opt= EI1_X2_grid[index_max_grid, :].reshape(1,-1)

    return x2_opt, x2_opt_value, ei1_x2_opt

class Ei_per_cost_optimize:


    def __init__(self, u0_x1, var0_x1, x1, Z, Xt, Yt):

        self.var0_x1= var0_x1
        L0_x1 = np.linalg.cholesky(var0_x1)
        y1 = u0_x1 + np.matmul(L0_x1, Z)
        self.Xt1 = np.append(Xt, x1, axis=0)
        self.Yt1 = np.append(Yt, y1, axis=0)


    def ei_pu_negative(self, x2, var0_x1, kernel, latent_cost_kernel, x1, Z, Xt, Yt_latent_cost,
                                 noise, model, latent_cost_model, f_best1, noise_cost):

        # print('function')
        x2 = x2.reshape(1, -1)

        u0_x2, var0_x2, sigma0_x2= get_mean_var_std(x2, model)

        self.u0_x2_cost = get_mean_of_cost(x2, latent_cost_model)

        self.u1_x2, self.var1_x2= get_one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise, var0_x1)
        self.sigma1_x2= np.sqrt(self.var1_x2)

        EI1_x2_pu =  ei_pu(self.sigma1_x2, self.u1_x2 , f_best1, self.u0_x2_cost)

        EI1_x2_pu = EI1_x2_pu.flatten()

        return -EI1_x2_pu


    def grad_ei_pu_negative(self, x2, var0_x1, kernel, latent_cost_kernel, x1, Z, Xt, Yt_latent_cost,
                                 noise, model, latent_cost_model, f_best1, noise_cost):

        # print('gradient')
        '''dEI1_x2'''
        x2 = x2.reshape(1, -1)

        EI1_x2_gradient = get_dEI_dx(f_best1, self.u1_x2, self.sigma1_x2, kernel, self.Xt1, self.Yt1, x2, noise)

        '''gradient of mean cost'''

        du0_x2_lat, dvar0_lat, _ = get_mean_variance_gradients_q1(latent_cost_kernel, Xt, Yt_latent_cost, x2, noise_cost)

        du0_x2_cost = du0_x2_lat * self.u0_x2_cost

        '''EI1_x2'''

        EI1_x2 = ei(self.sigma1_x2, self.u1_x2, f_best1)

        '''overall gradient term'''
        grad_pu = (self.u0_x2_cost * EI1_x2_gradient - du0_x2_cost * EI1_x2) / (self.u0_x2_cost ** 2)
        grad_pu = grad_pu.flatten()

        return -grad_pu


    def maximize_ei_pu(self, kernel, latent_cost_kernel, x1, Z, Xt, noise, model, latent_cost_model, domain,
                             num_inner_opt_restarts, grid_opt_in, D, f_best1, Yt_latent_cost, num_iter_max, noise_cost):

        '''scipy optimize'''
        if num_inner_opt_restarts > 0:
            lower = [];
            upper = []
            for i in range(len(domain)):
                lower.append(domain[i][0])
                upper.append(domain[i][1])
            b = Bounds(lb=lower, ub=upper)

            x2_opt_list_sci = np.zeros([num_inner_opt_restarts, D])
            x2_opt_value_list_sci = np.zeros([num_inner_opt_restarts, 1])

            for i in range(num_inner_opt_restarts):
                x20 = np.random.uniform(lower, upper, (1, D))
                fun_args= (self.var0_x1, kernel, latent_cost_kernel, x1, Z, Xt, Yt_latent_cost, noise, model, latent_cost_model, f_best1, noise_cost)

                result = (minimize(self.ei_pu_negative, x20, args= fun_args, bounds=b, method='L-BFGS-B', jac= self.grad_ei_pu_negative,
                          options={'maxiter':num_iter_max}))

                x2_cand = result['x'];
                x2_cand = x2_cand.reshape(1, -1)
                x2_cand_value = -result['fun'];
                x2_cand_value = x2_cand_value.reshape(1, -1)
                x2_cand_grad = -result['jac'];
                x2_cand_grad = x2_cand_grad.reshape(1, -1)

                x2_opt_list_sci[i, :] = x2_cand[0, :];
                x2_opt_value_list_sci[i, :] = x2_cand_value

            index_opt_sci = int(np.argmax(x2_opt_value_list_sci, axis=0))

            x2_opt_value_sci = x2_opt_value_list_sci[index_opt_sci, :].reshape(1, -1)

            x2_opt_sci = x2_opt_list_sci[index_opt_sci, :].reshape(1, -1)

            u0_x2_opt_sci, var0_x2_opt_sci = model.predict_f(x2_opt_sci);
            u0_x2_opt_sci = u0_x2_opt_sci.numpy();
            var0_x2_opt_sci = var0_x2_opt_sci.numpy()
            u1_x2, var1_x2 = get_one_step_mean_and_variance(x2_opt_sci, u0_x2_opt_sci, var0_x2_opt_sci, kernel, x1, Z, Xt,
                                                        noise, self.var0_x1)

            sigma1_x2 = np.sqrt(var1_x2)
            ei1_x2_opt_sci = ei(sigma1_x2, u1_x2, f_best1)

        '''grid optimize'''

        '''inner optimization with grid'''
        if D == 1 and grid_opt_in:
            disc = 101
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            X2_grid = x1_grid.reshape(-1, 1)

            x2_opt_grid, x2_opt_value_grid, ei1_x2_opt_grid = ei1_x2_pu_optimize_grid(X2_grid, kernel, noise, latent_cost_model,
                                                                                      self.Xt1, self.Yt1)

        if D == 2 and grid_opt_in:
            disc = 21
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
            # x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
            X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

            X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
            X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
            X2_grid = np.append(X1_flat, X2_flat, axis=1)

            x2_opt_grid, x2_opt_value_grid, ei1_x2_opt_grid =  ei1_x2_pu_optimize_grid(X2_grid, kernel, noise, latent_cost_model,
                                                                                      self.Xt1, self.Yt1)

        '''compare grid with sci optimum'''

        if (D == 2 or D == 1) and grid_opt_in and (num_inner_opt_restarts > 0):
            if x2_opt_value_grid > x2_opt_value_sci:
                x2_opt = x2_opt_grid;
                x2_opt_value = x2_opt_value_grid
                ei1_x2_opt = ei1_x2_opt_grid
            else:
                x2_opt = x2_opt_sci;
                x2_opt_value = x2_opt_value_sci
                ei1_x2_opt = ei1_x2_opt_sci
        elif num_inner_opt_restarts > 0:
            x2_opt = x2_opt_sci;
            x2_opt_value = x2_opt_value_sci
            ei1_x2_opt = ei1_x2_opt_sci
        else:
            x2_opt = x2_opt_grid
            x2_opt_value = x2_opt_value_grid
            ei1_x2_opt = ei1_x2_opt_grid

        return x2_opt_value, x2_opt, ei1_x2_opt


import sys
sys.path.append('./gp_gradients')
sys.path.append('..')

from gp_gradients import EI1_x2_per_cost_optimize

sys.path.append('../../functions')
from sine import *
sys.path.append('../../cost_functions')
from cos_1d import *
from exp_cos_1d import *

import time


def test_ei_per_cost_optimize():

    _, __, domain = sin_opt()
    D = 1; noise = 10 ** (-4)
    _, __, domain_cost= exp_cos_1d_opt()
    noise_cost = 10 ** (-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= sin(Xt)
    Yt_cost= exp_cos_1d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best = np.min(Yt, axis=0);
    num_iter_max = 100;


    x1_test = np.random.uniform(lower, upper, (1, D))
    Z= np.random.normal(0.0, 1.0, (1,1))
    u0_x1, var0_x1 = model.predict_f(x1_test);
    u0_x1 = u0_x1.numpy();
    var0_x1 = var0_x1.numpy()
    sigma0_x1 = np.sqrt(var0_x1)

    L0_x1 = np.linalg.cholesky(var0_x1)
    y1 = u0_x1 + np.matmul(L0_x1, Z)
    f_best1 = np.minimum(f_best, float(y1))


    '''update Xt and Yt according to outcome of Z'''
    Xt1 = np.append(Xt, x1_test, axis=0)
    Yt1 = np.append(Yt, y1, axis=0)

    num_inner_opt_restarts= 10; num_outer_opt_restarts=0; D=1;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;
    num_iter_inner_max = 100;



    time1_prev= time.clock()
    EI1_x2_per_cost_optimize(u0_x1, var0_x1, kernel, latent_cost_kernel, x1_test, Z, Xt, Yt, Yt_latent_cost,
                             noise, model, latent_cost_model, f_best, domain, num_inner_opt_restarts, grid_opt_in, D,
                             f_best1, num_iter_inner_max)
    time2_prev= time.clock()

    time1= time.clock()
    ei_pu_opt= Ei_per_cost_optimize(u0_x1, var0_x1, x1_test, Z, Xt, Yt)

    ei_pu_opt.maximize_ei_pu(kernel, latent_cost_kernel, x1_test, Z, Xt, noise, model, latent_cost_model, domain,
                             num_inner_opt_restarts, grid_opt_in, D, f_best1, Yt_latent_cost, num_iter_inner_max)
    time2= time.clock()

    print('unoptimized_py_code_time:{}, optimized_py_code_time:{}'.format((time2_prev-time1_prev),(time2- time1)))