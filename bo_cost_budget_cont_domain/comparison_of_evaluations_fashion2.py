import matplotlib.pyplot as plt
import numpy as np
from plots import plot_colormaps
from EI_pu_cont_domain import EI_pu_bo_cont_domain
from carbo import carbo_bo_cont_domain
from EI_cont_domain import ei_bo
from importance_cost_EI import imco_ei_bo

import sys
sys.path.append('../functions')
sys.path.append('../cost_functions')

'''objectives'''
from multi_opt_different_cost_1d import *
from multi_opt_different_cost_2d import *
from synthetic_2d import *
from synthetic_2d_multi2 import *
from synthetic_2d_multi_opt import *
from synthetic_2parts import *
'''costs'''
from multi_opt_different_cost_1d_cost import *
from multi_opt_different_cost_2d_cost import *
from synthetic_2d_cost import *
from synthetic_2d_multi2_cost import *
from synthetic_2d_multi_opt_cost import *
from synthetic_2parts_cost import *

sys.path.append('../HPO')
from keras_model import get_fashion2_domain,  initial_training_fashion2

D = 3;

objective_func = 'fashion2';
kernel= None; latent_cost_kernel= None
cost_function = None; x0= None
num_layer = None;num_dense= None;
num_init_train_samples = 5;
hyper_opt_per= 5

noise = 10 ** (-3);
noise_cost = 10 ** (-3)
domain = get_fashion2_domain()

num_layer = None; num_dense= None;
hyper_opt_per= 5

lower = [domain[i][0] for i in range(len(domain))];
upper = [domain[i][1] for i in range(len(domain))]


x0= np.random.uniform(lower, upper, (1,D))

random_restarts= 10
num_iter_max= 100

budget= 700;
grid= False
num_epoch= 1.7

X, Y, Y_cost = initial_training_fashion2(domain, num_init_train_samples, 1.7)

num_samples_uni = 2000;
num_samples_ei = 20

'''imco'''
loss_list_imco, Xt_imco, Yt_imco, model_imco, cost_list_imco, cum_cost_list_imco, latent_cost_model_imco, f_best_list_imco = \
    imco_ei_bo(D, objective_func, cost_function, None, None,
               domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max, grid,
               noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=None, num_dense=None,
               num_epoch= num_epoch, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per, plot_color=False,
               num_samples_uni=num_samples_uni, num_samples_ei=num_samples_ei)

'''carbo'''
loss_list_carbo, Xt_carbo, Yt_carbo, model_carbo, cost_list_carbo, cum_cost_list_carbo, latent_cost_kernel_carbo, f_best_list_carbo = \
    carbo_bo_cont_domain(D, objective_func, cost_function,None, None,
                         domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max,
                         grid,
                         noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=None,
                         num_dense=None, num_epoch=num_epoch, hyper_opt_per= hyper_opt_per, X_init=X, Y_init=Y,
                         Y_cost_init=Y_cost)

'''ei'''
loss_list_ei, Xt_ei, Yt_ei, model_ei, cost_list_ei, cum_cost_list_ei, latent_cost_kernel_ei, f_best_list_ei = \
    ei_bo(D, objective_func, cost_function, None, None,
          domain, kernel, budget, x0, latent_cost_kernel, random_restarts, grid, noise=noise,
          noise_cost=noise_cost, num_iter_max=num_iter_max, plot=False, plot_cost=False, num_layer=num_layer,
          num_dense=num_dense, num_epoch= num_epoch, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per)

'''ei_pu'''
loss_list_ei_pu, Xt_ei_pu, Yt_ei_pu, model_ei_pu, cost_list_ei_pu, cum_cost_list_ei_pu, latent_cost_kernel_pu, f_best_list_pu = \
    EI_pu_bo_cont_domain(D, objective_func, cost_function,None, None,
                         domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max,
                         grid,
                         noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=num_layer,
                         num_dense=num_dense, num_epoch= num_epoch, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per,
                         plot_color=False)

plt.figure()

plt.plot(np.squeeze(cum_cost_list_ei_pu), loss_list_ei_pu, label= 'ei_pu', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_ei_pu), loss_list_ei_pu, label= 'ei_pu', alpha= 0.5)

plt.plot(np.squeeze(cum_cost_list_ei), loss_list_ei, label= 'ei', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_ei), loss_list_ei, label= 'ei', alpha= 0.5)


plt.plot(np.squeeze(cum_cost_list_carbo),loss_list_carbo, label= 'carbo', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_carbo),loss_list_carbo, label= 'carbo', alpha= 0.5)

plt.plot(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)

plt.xlabel('cost'); plt.ylabel('loss'); plt.legend()
plt.show()


# plot_colormaps(Xt_carbo, disc, objective_func, cost_function, domain, 'carbo')
# plot_colormaps(Xt_ei, disc, objective_func, cost_function, domain, 'ei')
# plot_colormaps(Xt_ei_pu, disc, objective_func, cost_function, domain, 'ei_pu')
# plot_colormaps(Xt_imco,  disc, objective_func, cost_function, domain, 'imco')

