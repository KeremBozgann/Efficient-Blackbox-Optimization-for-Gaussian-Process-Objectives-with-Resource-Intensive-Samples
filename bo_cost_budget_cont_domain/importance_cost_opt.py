
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import gpflow as gp
from util import ei
from util import get_dEI_dx, get_mean_var_std,  get_grid, get_mean_and_var_cost,  get_mean_variance_gradients_q1
from ei_opt import Ei_opt

from util import ei_sampling

class imco_opt():

    def __init__(self, model, domain, f_best, num_samples_uni, num_samples_ei, latent_cost_model):



        X_ei, values= ei_sampling(num_samples_uni, num_samples_ei, domain, model, f_best)

        u_X_ei, var_X_ei, sigma_X_ei= get_mean_and_var_cost(X_ei, model)
        values= ei(sigma_X_ei, u_X_ei, f_best)
        u_cost_X_ei, _, _ = get_mean_and_var_cost(X_ei, latent_cost_model)
        self.mean_ei_pu= float(np.mean(values, axis=0)/np.mean(u_cost_X_ei, axis=0))

        # print('ei_avg_diff:{}, cost_prev:{}, dei_dcostself.dei_dcost{}'.format(dei, cost_prev, self.dei_dcost))
        # print('values_prev:{}, values:{}'.format(values_prev, values))
        print('avg ei-per_cost:{}\n'.format(self.mean_ei_pu))
    def imco_optimize_grid(self, X_grid, model, f_best, latent_cost_model):

        u_X_grid, var_X_grid, sigma_X2_grid = get_mean_var_std(X_grid, model)

        ei_X_grid = ei(sigma_X2_grid, u_X_grid, f_best)

        u_cost_X_grid, _, _ = get_mean_and_var_cost(X_grid, latent_cost_model)

        imco_grid = ei_X_grid -  self.mean_ei_pu* u_cost_X_grid

        index_max_grid = int(np.argmax(imco_grid, axis=0))
        x_opt = X_grid[index_max_grid, :].reshape(1, -1)
        x_opt_value = imco_grid[index_max_grid, :].reshape(1, -1)

        return x_opt, x_opt_value, imco_grid

    def imco_negative(self, x, model, f_best, latent_cost_model, kernel, Xt, Yt, noise,
                      latent_cost_kernel, Yt_cost):

        x = x.reshape(1, -1)

        self.u_x, self.var_x, self.sigma_x= get_mean_var_std(x, model)

        ei_x = ei(self.sigma_x, self.u_x, f_best)

        self.u_cost_x, _, _= get_mean_and_var_cost(x, latent_cost_model)

        imco= ei_x - self.mean_ei_pu * self.u_cost_x

        imco= imco.flatten()
        return -imco


    def grad_imco_negative(self, x,  model, f_best, latent_cost_model, kernel, Xt, Yt, noise,
                           latent_cost_kernel, Yt_cost):

        x = x.reshape(1, -1)

        '''ei_x and dEI_x'''

        grad_ei_x = get_dEI_dx(f_best, self.u_x, self.sigma_x, kernel, Xt, Yt, x, noise)

        Yt_latent_cost= np.log(Yt_cost)
        du_x_lat_cost, _, __= get_mean_variance_gradients_q1(latent_cost_kernel, Xt, Yt_latent_cost, x, noise)
        du_x_cost = du_x_lat_cost* self.u_cost_x
        grad_imco = grad_ei_x- self.mean_ei_pu*du_x_cost
        grad_imco= grad_imco.flatten()

        return -grad_imco


    def maximize_imco(self, kernel, Xt, noise, model, domain, num_opt_restarts, f_best,
                       grid, D, num_iter_max, Yt, latent_cost_model, latent_cost_kernel, Yt_cost):

        '''scipy optimize'''
        if num_opt_restarts > 0:
            lower = [];
            upper = []
            for i in range(len(domain)):
                lower.append(domain[i][0])
                upper.append(domain[i][1])
            b = Bounds(lb=lower, ub=upper)

            x_opt_list_sci = np.zeros([num_opt_restarts, D])
            x_opt_value_list_sci = np.zeros([num_opt_restarts, 1])

            for i in range(num_opt_restarts):
                x = np.random.uniform(lower, upper, (1, D))
                fun_args = (model, f_best, latent_cost_model, kernel, Xt, Yt, noise,
                      latent_cost_kernel, Yt_cost)

                result = (minimize(self.imco_negative, x, args=fun_args, bounds=b, method='L-BFGS-B',
                                   jac=self.grad_imco_negative, options={'maxiter': num_iter_max}))

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


        '''grid optimize'''

        '''inner optimization with grid'''
        if D == 1 and grid:
            disc = 101
            X_grid= get_grid(domain, disc)

            x_opt_grid, x_opt_value_grid, _ =self.imco_optimize_grid(X_grid, model, f_best, latent_cost_model,
                                                                  )


        if D == 2 and grid:
            disc = 21
            X_grid= get_grid(domain, disc)

            x_opt_grid, x_opt_value_grid, _ = self.imco_optimize_grid(X_grid, model, f_best, latent_cost_model,
                                                                  )
        '''compare grid with sci optimum'''

        if (D == 2 or D == 1) and grid and (num_opt_restarts > 0):
            if x_opt_value_grid > x_opt_value_sci:
                x_opt = x_opt_grid;
                x_opt_value = x_opt_value_grid

            else:
                x_opt = x_opt_sci;
                x_opt_value = x_opt_value_sci

        elif num_opt_restarts > 0:
            x_opt = x_opt_sci;
            x_opt_value = x_opt_value_sci

        else:
            x_opt = x_opt_grid
            x_opt_value = x_opt_value_grid


        return x_opt, x_opt_value
import sys
sys.path.append('../functions')

from sine import sin
import time
sys.path.append('../cost_functions')
from cos_1d import *
from exp_cos_1d import *

def test_imco_gradient():

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

    opt= imco_opt(kernel, Xt, noise, model, domain, random_restarts, f_best,
                                        grid, D, num_iter_max, Yt)

    ei_max_prev= 0.2; cost_prev= 0.5

    x_test_val = -opt.imco_negative(x_test, model, f_best, ei_max_prev, cost_prev, latent_cost_model, kernel, Xt, Yt,
                                    noise, latent_cost_kernel, Yt_cost)
    grad_x_test= - opt.grad_imco_negative(x_test, model, f_best, ei_max_prev, cost_prev, latent_cost_model, kernel, Xt, Yt,
                                          noise, latent_cost_kernel, Yt_cost)
    x_pl = x_test + 0.00001; x_mn = x_test - 0.00001;

    x_pl_val= -opt.imco_negative(x_pl, model, f_best, ei_max_prev, cost_prev, latent_cost_model, kernel, Xt, Yt,
                                 noise, latent_cost_kernel, Yt_cost)
    x_mn_val= -opt.imco_negative(x_mn, model, f_best, ei_max_prev, cost_prev, latent_cost_model, kernel, Xt, Yt,
                                 noise, latent_cost_kernel, Yt_cost)

    app_deriv= (( x_pl_val- x_test_val )/(x_pl- x_test)+\
                                    (x_test_val- x_mn_val)/(x_test- x_mn))/2

    print('\nanalytical_deriv:{}, app_deriv:{}'.format(grad_x_test, app_deriv))
