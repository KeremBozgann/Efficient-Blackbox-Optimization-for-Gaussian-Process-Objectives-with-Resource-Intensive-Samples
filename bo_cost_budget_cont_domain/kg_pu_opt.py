
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import gpflow as gp
from util import get_mean_var_std, get_mean_and_var_cost, get_one_step_mean_and_variance, dEI1_x2_dx1, get_mean_and_var_cost, \
                    get_mean_variance_gradients_q1

from ei_opt import Ei_opt

def get_model(Xt, Yt, kernel, noise):

    model = gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)
    return model

def get_cholesky_parameters(var):
    L= np.linalg.cholesky(var)
    L_inv= np.linalg.inv(L)
    L_inv_trans= np.transpose(L_inv)
    return L, L_inv, L_inv_trans

def get_second_step_params(u_x, var_x, z, Xt, Yt, x, kernel, noise):

    y_x = u_x + np.matmul(np.linalg.cholesky(var_x), np.atleast_2d(z))
    Xt1 = np.append(Xt, x, axis=0);
    Yt1 = np.append(Yt, y_x, axis=0)
    model2 = get_model(Xt1, Yt1, kernel, noise)
    f_best1 = float(np.min(Yt1, axis=0))

    return model2, Xt1, Yt1, y_x, f_best1

class Kgpc_optimize:


    def __init__(self, n):

        self.Z, self.W = np.polynomial.hermite.hermgauss(n)
        self.n= n
        pass

    def ei_kgpc_negative(self, x, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D):

        x = x.reshape(1, -1)

        opt1= Ei_opt()

        x1_opt, x1_opt_value= opt1.maximize_ei(kernel, Xt, noise, model, domain, num_inner_opt_restarts, f_best,
                                               grid_opt_in, D, num_iter_max, Yt)

        self.ei1= x1_opt_value.copy()

        #lists to be used by gradient function
        self.f_best1_list= np.empty([self.n, 1])
        self.x2_list= np.empty([self.n, D])
        self.y1_list= np.empty([self.n, 1])
        self.model2_list= []

        self.ei2_avg = 0
        for k in range(self.n):

            z= self.Z[k]; w= self.W[k]
            z*= np.sqrt(2)


            self.u_x, self.var_x, self.sigma_x= get_mean_var_std(x, model)

            model2, Xt1, Yt1, y_x, f_best1= get_second_step_params(self.u_x, self.var_x, z, Xt, Yt, x, kernel, noise)

            self.f_best1_list[k, 0]= f_best1
            self.y1_list[k, 0]= y_x

            self.model2_list.append(model2)

            opt2=  Ei_opt()
            x2_opt, x2_opt_value = opt2.maximize_ei(kernel, Xt1, noise, model2, domain, num_inner_opt_restarts, f_best1,
                                                    grid_opt_in, D, num_iter_max, Yt1)

            self.x2_list[k, :] = x2_opt[0, :]

            ei2_k= x2_opt_value*1/np.sqrt(np.pi)*w
            self.ei2_avg+= ei2_k

        # get the mean cost
        self.u_x_cost, self.u_x_lat_cost, self.var_x_lat_cost = get_mean_and_var_cost(x, latent_cost_model)

        kg_pu= (self.ei1- self.ei2_avg)/self.u_x_cost

        kg_pu= kg_pu.flatten()

        return -kg_pu


    def grad_ei_kgpc_negative(self, x, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D):

        x = x.reshape(1, -1)

        ei2_grad_avg = 0


        for k in range(self.n):
            z = self.Z[k];
            w = self.W[k]
            z *= np.sqrt(2)

            f_best1= self.f_best1_list[k, 0]
            y1= self.y1_list[k, 0]
            model2= self.model2_list[k]
            x2_opt= self.x2_list[k, :].reshape(1,-1)
            u_x2_opt, var_x2_opt, sigma_x2_opt= get_mean_var_std(x2_opt, model)
            # print('self.u_x, self.var_x, self.sigma_x', self.u_x, self.var_x, self.sigma_x)
            L_x, L_inv_x, L_inv_trans_x= get_cholesky_parameters(self.var_x)
            grad_ei_x2_x_k = dEI1_x2_dx1(f_best, f_best1, kernel, Xt, Yt, noise, x2_opt, x, D, z, u_x2_opt, var_x2_opt,
                                       L_x, L_inv_x, L_inv_trans_x, y1)


            grad_ei_x2_x_k *= 1 / np.sqrt(np.pi) * w

            ei2_grad_avg += grad_ei_x2_x_k

        u_x_cost, u_x_latent_cost, var_x_latent_cost= get_mean_and_var_cost(x, latent_cost_model)
        du_lat_cost_x, dvar_lat_cost_dx, _ = get_mean_variance_gradients_q1(latent_cost_kernel, Xt, Yt_latent_cost,
                                                                               x, noise_cost)


        grad_cost= du_lat_cost_x.copy()
        overall_gradient= -1/(u_x_cost)*(ei2_grad_avg)- grad_cost/(u_x_cost**2)*(self.ei1- self.ei2_avg)
        overall_gradient= overall_gradient.flatten()
        return -overall_gradient

    def maximize_kgpc(self, kernel, latent_cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
                             num_outer_opt_restarts, num_inner_opt_restarts, grid_opt_out, grid_opt_in, D, Yt_latent_cost, num_iter_max, noise_cost):

        '''scipy optimize'''
        if num_outer_opt_restarts > 0:
            lower = [];
            upper = []
            for i in range(len(domain)):
                lower.append(domain[i][0])
                upper.append(domain[i][1])
            b = Bounds(lb=lower, ub=upper)

            x_opt_list_sci = np.zeros([num_outer_opt_restarts, D])
            x_opt_value_list_sci = np.zeros([num_outer_opt_restarts, 1])

            for i in range(num_outer_opt_restarts):
                x0 = np.random.uniform(lower, upper, (1, D))
                print('outer_opt:{}'.format(i))

                fun_args= (model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, D)

                result = (minimize(self.ei_kgpc_negative, x0, args= fun_args, bounds=b, method='L-BFGS-B',
                                   jac= self.grad_ei_kgpc_negative,
                          options={'maxiter':num_iter_max}))

                x_cand = result['x'];
                x_cand = x_cand.reshape(1, -1)
                x_cand_value = -result['fun'];
                x_cand_value = x_cand_value.reshape(1, -1)
                x_cand_grad = -result['jac'];
                x_cand_grad = x_cand_grad.reshape(1, -1)

                x_opt_list_sci[i, :] = x_cand[0, :];
                x_opt_value_list_sci[i, :] = x_cand_value

            index_opt_sci = int(np.argmax(x_opt_value_list_sci, axis=0))

            x_opt_value_sci = x_opt_value_list_sci[index_opt_sci, :].reshape(1, -1)

            x_opt_sci = x_opt_list_sci[index_opt_sci, :].reshape(1, -1)

            # u_x_opt_sci, var_x_opt_sci, sigma_x_opt_sci = get_mean_var_std(x_opt_sci, model)

            # ei_x_opt_sci = ei(sigma_x_opt_sci,  u_x_opt_sci, f_best)

        '''grid optimize'''

        '''outer optimization with grid'''

        if grid_opt_out== True:

            if D==1:

                disc = 101
                x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
                X_grid = x1_grid.reshape(-1, 1)

                x_value_list_grid= np.zeros([disc, 1])

                for j in range(disc):
                    x_cand= X_grid[j,:].reshape(1,-1)

                    x_cand_value= -self.ei_kgpc_negative(x_cand, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)

                    x_cand_value= x_cand_value.reshape(1,-1)
                    x_value_list_grid[j, :]= x_cand_value[0,:]

                index_max_grid = int(np.argmax(x_value_list_grid, axis=0))
                x_opt_value_grid =x_value_list_grid[index_max_grid, :].reshape(1, -1)
                x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)

            if D == 2:

                disc = 21
                x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
                x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
                x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
                X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

                X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
                X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
                X_grid = np.append(X1_flat, X2_flat, axis=1)

                x_value_list_grid = np.zeros([disc**2, 1])


                for j in range(disc**2):
                    x_cand = X_grid[j, :].reshape(1, -1)
                    x_cand_value =-self.ei_kgpc_negative(x_cand, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D);

                    x_cand_value = x_cand_value.reshape(1, -1)
                    x_value_list_grid[j, :] = x_cand_value[0, :]

                index_max_grid = int(np.argmax(x_value_list_grid, axis=0))
                x_opt_value_grid =x_value_list_grid[index_max_grid, :].reshape(1, -1)
                x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)

        '''compare grid with sci optimum'''

        if (D == 2 or D == 1) and grid_opt_out and (num_outer_opt_restarts > 0):

            if x_opt_value_grid > x_opt_value_sci:
                x_opt = x_opt_grid;
                x_opt_value = x_opt_value_grid
            else:
                x_opt = x_opt_sci;
                x_opt_value = x_opt_value_sci
            return x_opt, x_opt_value, x_value_list_grid, X_grid

        elif grid_opt_out and (num_outer_opt_restarts > 0):

            x_opt = x_opt_sci;
            x_opt_value = x_opt_value_sci
            return x_opt, x_opt_value, None, None

        elif (not grid_opt_out) and (num_outer_opt_restarts > 0):

            x_opt = x_opt_sci;
            x_opt_value = x_opt_value_sci
            return x_opt, x_opt_value, None, None


        elif grid_opt_out and (not num_outer_opt_restarts > 0):

            x_opt = x_opt_grid
            x_opt_value = x_opt_value_grid

            return x_opt, x_opt_value, x_value_list_grid, X_grid


import sys
sys.path.append('./gp_gradients')
sys.path.append('..')

from gp_gradients import EI1_x2_per_cost_optimize

sys.path.append('../functions')
from sine import *
sys.path.append('../cost_functions')
from cos_1d import *
from exp_cos_1d import *

import time


sys.path.append('../functions')
sys.path.append('../cost_functions')
from exp_cos_2d import *
from branin import *

def test_ei_kg_pu():

    _, __, domain = branin_opt()
    D = 2; noise = 10 ** (-3)
    _, __, domain_cost= exp_cos_2d_opt()
    noise_cost = 10 ** (-3)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= branin(Xt)
    Yt_cost= exp_cos_2d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 0; num_outer_opt_restarts=0;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    n= 20
    x1_test = np.random.uniform(lower, upper, (1, D))

    opt= Ei_kgpc_optimize(n)
    # opt.maximize_ei_cepu(kernel, latent_cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
    #                          num_inner_opt_restarts, num_outer_opt_restarts, grid_opt_in, grid_opt_out, D, Yt_latent_cost, num_iter_max, noise_cost)

    x1_test_val= opt.ei_kgpc_negative(x1_test, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)

    x1_test_val*=-1
    '''analytical gradient'''
    x1_test_grad= -opt.grad_ei_kgpc_negative(x1_test, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)


    x11= x1_test.copy(); x11[0,0]= x11[0,0]+ 0.000001;
    x12= x1_test.copy(); x12[0,0] = x12[0,0]- 0.000001
    x21= x1_test.copy(); x21[0,1] = x21[0,1]+ 0.000001
    x22= x1_test.copy(); x22[0,1] = x22[0,1]- 0.000001


    x11_val = -opt.ei_kgpc_negative(x11, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)

    x12_val = -opt.ei_kgpc_negative(x12, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)
    x21_val = -opt.ei_kgpc_negative(x21, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)
    x22_val = -opt.ei_kgpc_negative(x22, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, Yt, domain, grid_opt_in, num_iter_max,num_inner_opt_restarts, D)

    grad_app_dir1 = ((x11_val -x1_test_val) / (x11[0, 0] - x1_test[0, 0]) + (x1_test_val - x12_val) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2 = ((x21_val - x1_test_val) / (x21[0, 1] - x1_test[0, 1]) + (x1_test_val - x22_val) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app = np.append(grad_app_dir1, grad_app_dir2, axis=0)

    print('analytical_grad: \n{}, \napp_grad:\n{}'.format(x1_test_grad, grad_app))
