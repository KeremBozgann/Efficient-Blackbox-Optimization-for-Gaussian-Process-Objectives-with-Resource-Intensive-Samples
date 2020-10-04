import matplotlib.pyplot as plt
import numpy as np
from plots import plot_colormaps
from EI_pu_cont_domain import EI_pu_bo_cont_domain
from carbo import carbo_bo_cont_domain
from EI_cont_domain import ei_bo
from ei_weighted_ei_pu import  eiw_eipu_bo
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
from griewank_2d import *
from griewank_3d import *

'''costs'''
from multi_opt_different_cost_1d_cost import *
from multi_opt_different_cost_2d_cost import *
from synthetic_2d_cost import *
from synthetic_2d_multi2_cost import *
from synthetic_2d_multi_opt_cost import *
from synthetic_2parts_cost import *
from branin_res_cost import *
from griewank_3d_cost import *

disc =5
D = 3
noise = 10 ** (-3);
noise_cost = 10 ** (-3);

objective_func = griewank_3d;
y_true_opt, x_true_opt, domain =griewank_3d_opt()
X_kern, Y_kern = griewank_3d_plots(disc)
model, kernel =griewank_3d_find_best_suited_kernel(X_kern, Y_kern, noise=noise)

X_cost_kern, Y_cost_kern = griewank_3d_cost_plots(disc)
latent_cost_model, latent_cost_kernel =griewank_3d_cost_find_best_suited_kernel(X_cost_kern, Y_cost_kern, noise=noise_cost)
cost_function = griewank_3d_cost

num_layer = None; num_dense= None;
hyper_opt_per= 5

lower = [domain[i][0] for i in range(len(domain))];
upper = [domain[i][1] for i in range(len(domain))]


x0= np.random.uniform(lower, upper, (1,D))

random_restarts= 10
num_iter_max= 100

budget= 100;
grid= False

X= None; Y= None; Y_cost= None

num_lthc_samples = 3000;
num_ei_samples = 500

'''carbo'''
loss_list_carbo, Xt_carbo, Yt_carbo, model_carbo, cost_list_carbo, cum_cost_list_carbo, latent_cost_kernel_carbo, f_best_list_carbo = \
    carbo_bo_cont_domain(D, objective_func, cost_function, y_true_opt, x_true_opt,
                         domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max,
                         grid,
                         noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=None,
                         num_dense=None, num_epoch=None, hyper_opt_per=hyper_opt_per, X_init=X, Y_init=Y,
                         Y_cost_init=Y_cost)

'''eiw_eipu'''
loss_list_eiw_eipu, Xt_eiw_eipu, Yt_eiw_eipu, model_eiw_eipu, cost_list_eiw_eipu, cum_cost_list_eiw_eipu, \
latent_cost_model_eiw_eipu, f_best_list_eiw_eipu = \
    eiw_eipu_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max, grid,
                noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=None,
                num_dense=None,
                num_epoch= None, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=hyper_opt_per, plot_color=False,
                num_lthc_samples=num_lthc_samples, num_ei_samples=num_ei_samples, sampling_method='random', cut_below_avg= False)

'''imco'''
loss_list_imco, Xt_imco, Yt_imco, model_imco, cost_list_imco, cum_cost_list_imco, latent_cost_model_imco, f_best_list_imco = \
    imco_ei_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
               domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max, grid,
               noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=None, num_dense=None,
               num_epoch=None, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per= hyper_opt_per, plot_color=False,
               num_samples_uni= 1000, num_samples_ei= 100)


'''ei'''
loss_list_ei, Xt_ei, Yt_ei, model_ei, cost_list_ei, cum_cost_list_ei, latent_cost_kernel_ei, f_best_list_ei = \
    ei_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
          domain, kernel, budget, x0, latent_cost_kernel, random_restarts, grid, noise=noise,
          noise_cost=noise_cost, num_iter_max=num_iter_max, plot=False, plot_cost=False, num_layer=num_layer,
          num_dense=num_dense, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per)

'''ei_pu'''
loss_list_ei_pu, Xt_ei_pu, Yt_ei_pu, model_ei_pu, cost_list_ei_pu, cum_cost_list_ei_pu, latent_cost_kernel_pu, f_best_list_pu = \
    EI_pu_bo_cont_domain(D, objective_func, cost_function, y_true_opt, x_true_opt,
                         domain, kernel, budget, x0, latent_cost_kernel, random_restarts, num_iter_max,
                         grid,
                         noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=num_layer,
                         num_dense=num_dense, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per,
                         plot_color=False)

plt.figure()

plt.plot(np.squeeze(cum_cost_list_ei_pu), np.array(loss_list_ei_pu), label= 'ei_pu', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_ei_pu), np.array(loss_list_ei_pu), label= 'ei_pu', alpha= 0.5)

plt.plot(np.squeeze(cum_cost_list_ei),np.array(loss_list_ei), label= 'ei', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_ei), np.array(loss_list_ei), label= 'ei', alpha= 0.5)


plt.plot(np.squeeze(cum_cost_list_carbo), np.array(loss_list_carbo), label= 'carbo', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_carbo),np.array(loss_list_carbo), label= 'carbo', alpha= 0.5)

plt.plot(np.squeeze(cum_cost_list_eiw_eipu), np.array(loss_list_eiw_eipu), label= 'eiw_eipu', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_eiw_eipu),np.array(loss_list_eiw_eipu), label= 'eiw_eipu', alpha= 0.5)

plt.plot(np.squeeze(cum_cost_list_imco), np.array(loss_list_imco), label= 'imco', alpha= 0.5)
plt.scatter(np.squeeze(cum_cost_list_imco),np.array(loss_list_imco), label= 'imco', alpha= 0.5)

# plt.plot(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)

plt.xlabel('cost'); plt.ylabel('loss'); plt.legend()
plt.show()

#
# plt.figure()
#
# plt.plot(np.squeeze(cum_cost_list_ei_pu), np.log10(loss_list_ei_pu), label= 'ei_pu', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_ei_pu), np.log10(loss_list_ei_pu), label= 'ei_pu', alpha= 0.5)
#
# plt.plot(np.squeeze(cum_cost_list_ei), np.log10(loss_list_ei), label= 'ei', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_ei), np.log10(loss_list_ei), label= 'ei', alpha= 0.5)
#
#
# plt.plot(np.squeeze(cum_cost_list_carbo), np.log10(loss_list_carbo), label= 'carbo', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_carbo), np.log10(loss_list_carbo), label= 'carbo', alpha= 0.5)
#
# plt.plot(np.squeeze(cum_cost_list_eiw_eipu), np.log10(loss_list_eiw_eipu), label= 'eiw_eipu', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_eiw_eipu),np.log10(loss_list_eiw_eipu), label= 'eiw_eipu', alpha= 0.5)
#
# # plt.plot(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)
# # plt.scatter(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)
#
# plt.xlabel('cost'); plt.ylabel('loss'); plt.legend()
# plt.show()


# plt.figure()
#
# plt.plot(np.squeeze(cum_cost_list_ei_pu), np.squeeze(np.log10(np.array(f_best_list_pu)-y_true_opt)), label= 'ei_pu', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_ei_pu), np.squeeze(np.log10(np.array(f_best_list_pu)-y_true_opt)), label= 'ei_pu', alpha= 0.5)
#
# plt.plot(np.squeeze(cum_cost_list_ei), np.squeeze(np.log10(np.array(f_best_list_ei)-y_true_opt)), label= 'ei', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_ei), np.squeeze(np.log10(np.array(f_best_list_ei)-y_true_opt)), label= 'ei', alpha= 0.5)
#
#
# # plt.plot(np.squeeze(cum_cost_list_carbo), np.squeeze(np.log10(np.array(f_best_list_carbo)-y_true_opt)), label= 'carbo', alpha= 0.5)
# # plt.scatter(np.squeeze(cum_cost_list_carbo),  np.squeeze(np.log10(np.array(f_best_list_carbo)-y_true_opt)), label= 'carbo', alpha= 0.5)
#
# plt.plot(np.squeeze(cum_cost_list_eiw_eipu),  np.squeeze(np.log10(np.array(f_best_list_eiw_eipu)-y_true_opt)), label= 'eiw_eipu', alpha= 0.5)
# plt.scatter(np.squeeze(cum_cost_list_eiw_eipu), np.squeeze(np.log10(np.array(f_best_list_eiw_eipu)-y_true_opt)), label= 'eiw_eipu', alpha= 0.5)
#
# # plt.plot(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)
# # plt.scatter(np.squeeze(cum_cost_list_imco), loss_list_imco, label= 'imco', alpha= 0.5)
#
# plt.xlabel('cost'); plt.ylabel('loss'); plt.legend()
# plt.show()

plot_colormaps(Xt_carbo, disc, objective_func, cost_function, domain, 'carbo')
plot_colormaps(Xt_ei, disc, objective_func, cost_function, domain, 'ei')
plot_colormaps(Xt_ei_pu, disc, objective_func, cost_function, domain, 'ei_pu')
plot_colormaps(Xt_eiw_eipu,  disc, objective_func, cost_function, domain, 'eiw_eipu')

