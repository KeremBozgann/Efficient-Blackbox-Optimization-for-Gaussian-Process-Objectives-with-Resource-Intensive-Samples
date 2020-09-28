
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import gpflow as gp
from util import ei
from util import get_dEI_dx, get_mean_var_std,  get_grid
from ei_opt import Ei_opt
from EI_pu_opt import EI_pu_optimize_with_gradient



def get_optimum_ei(kernel, Xt, noise, model, domain, num_inner_opt_restarts, f_best, grid, D, num_iter_max, Yt):

    opt_ei = Ei_opt()
    x_opt_ei, x_opt_ei_value = opt_ei.maximize_ei(kernel, Xt, noise, model, domain, num_inner_opt_restarts,
                                                  f_best, grid, D, num_iter_max, Yt)

    return x_opt_ei, x_opt_ei_value

#
# class Ei_proximity_search():
#
#     def __init__(self):
#         pass



def ei_proximity_search_opt(kernel, Xt, noise, model, domain, num_opt_restarts, f_best,grid, D, num_iter_max, Yt,
                     latent_cost_model, Yt_cost, latent_cost_kernel):

    x_opt_ei, x_opt_ei_value= get_optimum_ei(kernel, Xt, noise, model, domain,  num_opt_restarts,
                                              f_best,grid, D, num_iter_max, Yt)

    print('x_opt_ei:{}, x_opt_ei_value:{}'.format(x_opt_ei, x_opt_ei_value))
    search_range= np.empty([len(domain), 1])

    for i in range(len(domain)):
        search_range[i,0] = (domain[i][1] - domain[i][0])/20
    print('search range:{}'.format(search_range))
    domain_proximity= []

    for i in range(len(domain)):
        loweri= (x_opt_ei[0, i]- search_range[i,0])
        upperi= (x_opt_ei[0, i]+ search_range[i,0])
        if loweri<domain[i][0]:
            loweri= domain[i][0]

        if upperi>domain[i][1]:
            upperi= domain[i][1]

        domain_proximity.append([loweri, upperi])

    print('domain proximity:{}'.format(domain_proximity))
    x_opt_ei_pu, x_opt_ei_pu_value= EI_pu_optimize_with_gradient(domain_proximity, model, latent_cost_model,
                                         num_opt_restarts, num_iter_max, Xt, Yt, Yt_cost, f_best,
                                                     kernel, latent_cost_kernel, noise, D, grid)


    return x_opt_ei_pu, x_opt_ei_pu_value
