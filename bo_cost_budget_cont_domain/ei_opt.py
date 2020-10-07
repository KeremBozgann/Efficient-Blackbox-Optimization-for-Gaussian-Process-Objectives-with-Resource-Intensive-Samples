
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import gpflow as gp
from util import ei
from util import get_dEI_dx, get_mean_var_std,  get_grid

def ei_optimize_grid(X_grid, model, f_best):


    u_X_grid, var_X_grid, sigma_X2_grid= get_mean_var_std(X_grid, model)


    ei_X_grid= ei(sigma_X2_grid, u_X_grid, f_best)

    index_max_grid= int(np.argmax(ei_X_grid, axis=0))
    x_opt= X_grid[index_max_grid, :].reshape(1,-1)
    x_opt_value= ei_X_grid[index_max_grid, :].reshape(1,-1)

    return x_opt, x_opt_value

class Ei_opt():

    def __init__(self):
        pass

    def ei_negative(self, x, model, f_best, kernel, Xt, Yt, noise):

        x = x.reshape(1, -1)

        self.u_x, self.var_x, self.sigma_x = get_mean_var_std(x, model)

        ei_x= ei(self.sigma_x, self.u_x, f_best)

        ei_x= ei_x.flatten()
        return -ei_x


    def grad_ei_negative(self, x, model, f_best, kernel, Xt, Yt, noise):

        x = x.reshape(1, -1)

        '''ei_x and dEI_x'''

        grad_ei_x = get_dEI_dx(f_best, self.u_x, self.sigma_x, kernel, Xt, Yt, x, noise)

        grad_ei_x= grad_ei_x.flatten()

        return -grad_ei_x


    def maximize_ei(self, kernel, Xt, noise, model, domain, num_opt_restarts, f_best,
                       grid, D, num_iter_max, Yt):

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
                fun_args = (model, f_best, kernel, Xt, Yt, noise)

                result = (minimize(self.ei_negative, x, args=fun_args, bounds=b, method='L-BFGS-B',
                                   jac=self.grad_ei_negative, options={'maxiter': num_iter_max}))

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

            x_opt_grid, x_opt_value_grid = ei_optimize_grid(X_grid, model, f_best)


        if D == 2 and grid:
            disc = 21
            X_grid= get_grid(domain, disc)

            x_opt_grid, x_opt_value_grid = ei_optimize_grid(X_grid, model, f_best)
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