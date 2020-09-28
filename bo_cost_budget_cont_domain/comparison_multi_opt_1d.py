import numpy as np
import gpflow as gp

import importlib as imp
import Acquisitions
imp.reload(Acquisitions)


from EI_pu_cont_domain import EI_pu_bo_cont_domain
from carbo import carbo_bo_cont_domain
# from ei_pucla import ei_pucla_bo
from p_ei_pu import ei_ppu_bo
# from ei_pucla_v2 import ei_pucla_v2_bo
# from CATS import CATS_bo_cont_domain
from EI_cont_domain import ei_bo
from importance_cost_EI import imco_ei_bo

import sys
sys.path.append('../')
from gp_sample_save_to_dataset import *
from plots import *

sys.path.append('../cost_functions')
sys.path.append('../functions')

'''objective functions'''
from six_hump_camel import *
from branin import *
from sine import *
from quadratic_2d import *
from multi_opt_different_cost_1d import *
from multi_opt_different_cost_2d import *
from synthetic_2d_cost import *

'''cost functions'''
from exp_cos_1d import *
from exp_cos_2d import *
from quadratic_2d_cost import *
from multi_opt_different_cost_1d_cost import *
from multi_opt_different_cost_2d_cost import *
from synthetic_2d import *

from keras_model import initial_training_cifar
from keras_model import initial_training_fashion
from keras_model import get_fashion_domain
from keras_model import get_cifar_domain
from ei_proximity_search import ei_proximity_search_bo

from util_results import *

'''folder and file name for writing results'''
folder= '14_09_2020'
exp_name= 'exp1'
hf_name= 'exp1'

'''load data'''

sys.path.append('../cost_functions')
sys.path.append('../functions')

'''dimension dependent assignments'''
# disc= 101

disc =101
D = 1
noise = 10 ** (-3);
noise_cost = 10 ** (-3);

objective_func =multi_opt_1d;
y_true_opt, x_true_opt, domain = multi_opt_1d_opt()
X_kern, Y_kern = multi_opt_1d_plots(disc, plot=False)
model, kernel = multi_opt_1d_find_best_suited_kernel(X_kern, Y_kern, noise=noise)

X_cost_kern, Y_cost_kern = multi_opt_1d_cost_plots(disc, False)
latent_cost_model, latent_cost_kernel =multi_opt_1d_cost_find_best_suited_kernel(X_cost_kern, Y_cost_kern, noise=noise_cost)
cost_function = multi_opt_1d_cost

num_layer = None; num_dense= None;
hyper_opt_per= False

lower = [domain[i][0] for i in range(len(domain))];
upper = [domain[i][1] for i in range(len(domain))]

'''imco parameters'''
random_restarts_imco= 10
num_iter_max_imco= 100
num_samples_uni_imco= 1000
num_samples_ei_imco= 10
grid_imco= False

'''ei_pu parameters'''
random_restarts_ei_pu= 10
num_iter_max_ei_pu= 100
grid_pu= False


'''carbo parameters'''
random_restarts_carbo= 10
num_iter_max_carbo= 100
grid_carbo= False

# '''cats parameters'''
# grid_opt_in= True; grid_opt_out= True
# num_inner_opt_restarts= 0; num_outer_opt_restarts= 0
# num_iter_max_cats= 100

# '''pucla parameters'''
# grid_opt_in= True; grid_opt_out= True
# num_inner_opt_restarts= 0; num_outer_opt_restarts= 0
# num_iter_max_pucla= 100

# '''pucla_v2 parameters'''
# grid_opt_in_v2= True; grid_opt_out_v2= True
# num_inner_opt_restarts_v2= 0; num_outer_opt_restarts_v2= 0
# num_iter_max_pucla_v2= 100

# '''ei_ppu parameters'''
# random_restarts_ei_ppu= 10
# num_iter_max_ei_ppu= 100
# grid_ppu= False

'''ei parameters'''
random_restarts_ei= 10
num_iter_max_ei= 100
grid_ei= False

# '''ei proximity search parameters'''
# random_restarts_prox= 10
# num_iter_max_prox= 100
# grid_prox= False

budget= 100; total_iterations= 10
num_init_train_samples= 5

cost_grid= np.linspace(0, 3/2*budget, 20)

plot_loss_at_each_iteration= False

count_ei= np.zeros(cost_grid.shape); loss_ei= np.zeros(cost_grid.shape); f_best_ei= np.zeros(cost_grid.shape)
count_ei_pu= np.zeros(cost_grid.shape); loss_ei_pu= np.zeros(cost_grid.shape); f_best_pu= np.zeros(cost_grid.shape)
count_carbo= np.zeros(cost_grid.shape); loss_carbo= np.zeros(cost_grid.shape); f_best_carbo= np.zeros(cost_grid.shape)
# count_cats= np.zeros(cost_grid.shape); loss_cats= np.zeros(cost_grid.shape);
# count_pucla= np.zeros(cost_grid.shape); loss_pucla= np.zeros(cost_grid.shape);
# count_pucla_v2= np.zeros(cost_grid.shape); loss_pucla_v2= np.zeros(cost_grid.shape);
# count_ppu= np.zeros(cost_grid.shape); loss_ppu= np.zeros(cost_grid.shape); f_best_ppu= np.zeros(cost_grid.shape)
# count_prox= np.zeros(cost_grid.shape); loss_prox= np.zeros(cost_grid.shape); f_best_prox= np.zeros(cost_grid.shape)
count_imco= np.zeros(cost_grid.shape); loss_imcox= np.zeros(cost_grid.shape); f_best_imco= np.zeros(cost_grid.shape)

for i in range(total_iterations):
    print('iteration: ', i)
    # kernel, latent_cost_kernel = find_best_suited_gp_kernels(X, Y, Y_cost, noise, noise_cost)

    if objective_func=='fashion':
        x0= None
        X, Y, Y_cost = initial_training_fashion(domain,  num_init_train_samples, num_layer)

    elif objective_func=='cifar':
        x0= None
        X, Y, Y_cost = initial_training_cifar(domain, num_init_train_samples, num_layer, num_dense)

    else:
        X = None; Y = None; Y_cost = None
        x0 = np.random.uniform(lower, upper, (1, D))

    #
    # '''ei'''
    # loss_list_EI, Xt_EI, Yt_EI, model_EI, cost_list_EI, cum_cost_list_EI= EI_bo_cont_domain(input_dimension, output_dimension,
    #                     cost_dimension, objective_func, y_true_opt, x_true_opt, domain, kernel, budget, x0,
    #                                 cost_kernel, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False)
    #
    # loss_ei, count_ei= add_loss(loss_ei, count_ei, loss_list_EI,   np.atleast_1d(np.array(cum_cost_list_EI).squeeze()),
    #                             cost_grid)

    print('entering imco')
    '''imco'''
    loss_list_imco, Xt_imco, Yt_imco, model_imco, cost_list_imco, cum_cost_list_imco, latent_cost_model_imco, f_best_list_imco = \
        imco_ei_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                   domain, kernel, budget, x0, latent_cost_kernel, random_restarts_imco, num_iter_max_imco, grid_imco,
                   noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer=None, num_dense=None,
                   num_epoch=None, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=False, plot_color=False,
                   num_samples_uni=num_samples_uni_imco, num_samples_ei=num_samples_ei_imco)

    loss_list_imco = np.log10(loss_list_imco)
    loss_imco, count_imco, f_best_imco = add_loss(loss_imco, count_imco
                                   ,loss_list_imco, np.atleast_1d(np.array(cum_cost_list_imco).squeeze()),
                                                     f_best_list_imco, f_best_imco, cost_grid)

    print('entering carbo')
    '''carbo'''
    loss_list_carbo, Xt_carbo, Yt_carbo, model_carbo, cost_list_carbo, cum_cost_list_carbo, latent_cost_kernel_carbo,f_best_list_carbo = \
        carbo_bo_cont_domain(D, objective_func, cost_function, y_true_opt, x_true_opt,
                             domain, kernel, budget, x0, latent_cost_kernel, random_restarts_carbo, num_iter_max_carbo, grid_carbo,
                             noise= noise, noise_cost= noise_cost, plot= False, plot_cost=False, num_layer=None,
                             num_dense=None, num_epoch= None, hyper_opt_per=False, X_init=None, Y_init=None,
                             Y_cost_init=None)

    loss_list_carbo = np.log10(loss_list_carbo)
    loss_carbo, count_carbo, f_best_carbo = add_loss(loss_carbo, count_carbo
                                   ,loss_list_carbo, np.atleast_1d(np.array(cum_cost_list_carbo).squeeze()),
                                                     f_best_list_carbo, f_best_carbo, cost_grid)

    # print('entering proximity search')
    # '''ei proximity search'''
    # loss_list_prox, Xt_prox, Yt_prox, model_prox, cost_list_prox, cum_cost_list_prox, latent_cost_kernel_prox, f_best_list_prox= \
    #     ei_proximity_search_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
    #                        domain, kernel, budget, x0, latent_cost_kernel, random_restarts_prox, num_iter_max_prox, grid_prox,
    #                        noise=noise, noise_cost=noise_cost, plot=False, plot_cost=False, num_layer= num_layer,
    #                        num_dense= num_dense, X_init=X, Y_init=Y, Y_cost_init= Y_cost, hyper_opt_per=hyper_opt_per)
    #
    # loss_list_prox = np.log10(loss_list_prox)
    # loss_prox, count_prox, f_best_prox = add_loss(loss_prox, count_prox
    #                                    ,loss_list_prox, np.atleast_1d(np.array(cum_cost_list_prox).squeeze()),
    #                                               f_best_list_prox, f_best_prox, cost_grid)
    #
    #
    #
    # print('entering ei_ppu')
    # '''ei_ppu'''
    # loss_list_ppu, Xt_ppu, Yt_ppu, model_ppu, cost_list_ppu, cum_cost_list_ppu, latent_cost_kernel_ppu, f_best_list_ppu= \
    #     ei_ppu_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
    #               domain, kernel, budget, x0, latent_cost_kernel, random_restarts_ei_ppu, grid_ppu, noise=noise,
    #               noise_cost=noise_cost, num_iter_max= num_iter_max_ei_ppu, plot=False, plot_cost=False, num_layer=num_layer,
    #               num_dense= num_dense, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per)
    #
    # loss_list_ppu = np.log10(loss_list_ppu)
    # loss_ppu, count_ppu, f_best_ppu = add_loss(loss_ppu, count_ppu, loss_list_ppu, np.atleast_1d(np.array(cum_cost_list_ppu).squeeze()),
    #                                f_best_list_ppu, f_best_ppu, cost_grid)


    # '''pucla'''
    # loss_list_pucla, Xt_pucla, Yt_pucla, model_pucla, cost_list_pucla, cum_cost_list_pucla= \
    #     ei_pucla_bo(input_dimension, output_dimension, cost_dimension, objective_func, cost_function, y_true_opt, x_true_opt,
    #                         domain, kernel, budget, x0, cost_kernel, num_outer_opt_restarts, num_inner_opt_restarts,
    #                         grid_opt_in, grid_opt_out, noise= noise, noise_cost= noise_cost, num_iter_max= num_iter_max_pucla,
    #                         plot=False, plot_cost=False)
    # loss_list_pucla = np.log10(loss_list_pucla)
    # loss_pucla, count_pucla = add_loss(loss_pucla, count_pucla
    #                                ,loss_list_pucla, np.atleast_1d(np.array(cum_cost_list_pucla).squeeze()), cost_grid)

    # '''pucla_v2'''
    # loss_list_pucla_v2, Xt_pucla_v2, Yt_pucla_v2, model_pucla_v2, cost_list_pucla_v2, cum_cost_list_pucla_v2= \
    #     ei_pucla_v2_bo(input_dimension, output_dimension, cost_dimension, objective_func, cost_function, y_true_opt, x_true_opt,
    #                         domain, kernel, budget, x0, cost_kernel, num_outer_opt_restarts, num_inner_opt_restarts,
    #                         grid_opt_in, grid_opt_out, noise= noise, noise_cost= noise_cost, num_iter_max= num_iter_max_pucla,
    #                         plot=False, plot_cost=False)
    # loss_list_pucla_v2 = np.log10(loss_list_pucla_v2)
    # loss_pucla_v2, count_pucla_v2 = add_loss(loss_pucla_v2, count_pucla_v2
    #                                ,loss_list_pucla_v2, np.atleast_1d(np.array(cum_cost_list_pucla_v2).squeeze()), cost_grid)

    # '''pucla'''
    # loss_list_pucla, Xt_pucla, Yt_pucla, model_pucla, cost_list_pucla, cum_cost_list_pucla= \
    #     ei_pucla_bo(input_dimension, output_dimension, cost_dimension, objective_func, cost_function, y_true_opt, x_true_opt,
    #                         domain, kernel, budget, x0, cost_kernel, num_outer_opt_restarts, num_inner_opt_restarts,
    #                         grid_opt_in, grid_opt_out, noise= noise, noise_cost= noise_cost, num_iter_max= num_iter_max_pucla,
    #                         plot=False, plot_cost=False)
    # loss_list_pucla = np.log10(loss_list_pucla)
    # loss_pucla, count_pucla = add_loss(loss_pucla, count_pucla
    #                                ,loss_list_pucla, np.atleast_1d(np.array(cum_cost_list_pucla).squeeze()), cost_grid)

    print('entering ei_pu')
    '''ei_pu'''
    loss_list_ei_pu, Xt_ei_pu, Yt_ei_pu,model_ei_pu, cost_list_ei_pu, cum_cost_list_ei_pu, latent_cost_kernel_pu, f_best_list_pu = \
        EI_pu_bo_cont_domain(D, objective_func, cost_function, y_true_opt, x_true_opt,
                             domain, kernel, budget, x0, latent_cost_kernel, random_restarts_ei_pu, num_iter_max_ei_pu, grid_pu,
                             noise= noise, noise_cost= noise_cost, plot=False, plot_cost=False, num_layer=num_layer,
                             num_dense= num_dense, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per, plot_color= False)

    loss_list_ei_pu= np.log10(loss_list_ei_pu)
    loss_ei_pu, count_ei_pu, f_best_pu = add_loss(loss_ei_pu, count_ei_pu
                                   ,loss_list_ei_pu, np.atleast_1d(np.array(cum_cost_list_ei_pu).squeeze()),
                                                  f_best_list_pu, f_best_pu, cost_grid)

    print('entering ei')
    '''ei'''
    loss_list_ei, Xt_ei, Yt_ei, model_ei, coss_list_ei, cum_cost_list_ei, latent_cost_kernel_ei, f_best_list_ei = \
        ei_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
              domain, kernel, budget, x0, latent_cost_kernel, random_restarts_ei, grid_ei, noise=noise,
              noise_cost=noise_cost, num_iter_max= num_iter_max_ei, plot=False, plot_cost=False, num_layer=num_layer,
              num_dense= num_dense, X_init=X, Y_init=Y, Y_cost_init=Y_cost, hyper_opt_per=hyper_opt_per)

    loss_list_ei = np.log10(loss_list_ei)
    loss_ei, count_ei, f_best_ei = add_loss(loss_ei, count_ei
                                       , loss_list_ei, np.atleast_1d(np.array(cum_cost_list_ei).squeeze()), f_best_list_ei,
                                      f_best_ei, cost_grid)
    #
    # '''cats'''
    # loss_list_cats, Xt_cats, Yt_cats, model_cats, cost_list_cats, cum_cost_list_cats= \
    #     CATS_bo_cont_domain(input_dimension, output_dimension, cost_dimension, objective_func, cost_function, y_true_opt, x_true_opt,
    #                         domain, kernel, budget, x0, cost_kernel, num_outer_opt_restarts, num_inner_opt_restarts,
    #                         grid_opt_in, grid_opt_out, noise= noise, noise_cost= noise_cost, num_iter_max= num_iter_max_cats,
    #                         plot=False, plot_cost=False)
    #
    # loss_cats, count_cats = add_loss(loss_cats, count_cats
    #                                ,loss_list_cats, np.atleast_1d(np.array(cum_cost_list_cats).squeeze()), cost_grid)

    loss_and_count_dict = {'ei_pu': [loss_ei_pu, count_ei_pu, cost_grid, f_best_pu],
                           'carbo': [loss_carbo, count_carbo, cost_grid, f_best_carbo],
                            'ei': [loss_ei, count_ei, cost_grid, f_best_ei],
                            'imco': [loss_imco, count_imco, cost_grid, f_best_imco],
                           }

    save_to_hf(loss_and_count_dict, folder, hf_name, i)

    if plot_loss_at_each_iteration:

        plt.figure()
        plt.title('loss vs cost')
        plt.plot(np.squeeze(cum_cost_list_ei_pu), np.array(loss_list_ei_pu), label= 'ei_per_cost', color= 'blue')
        plt.scatter(np.squeeze(cum_cost_list_ei_pu), np.array(loss_list_ei_pu), label= 'ei_per_cost', color= 'blue')
        plt.plot(np.squeeze(cum_cost_list_ei_pu), np.array(loss_list_ei_pu), label= 'ei_per_cost', color= 'blue')
        plt.scatter(np.squeeze(cum_cost_list_ei_pu), np.array(loss_list_ei_pu), label= 'ei_per_cost', color= 'blue')

        # plt.plot(np.squeeze(cum_cost_list_EI), np.array(loss_list_EI), label= 'ei', color= 'red')
        # plt.scatter(np.squeeze(cum_cost_list_EI), np.array(loss_list_EI), label= 'ei', color= 'red')

        plt.legend()
        plt.xlabel('cost'); plt.ylabel('loss')
        plt.show()




loss_and_count_dict= {'ei_pu':[loss_ei_pu, count_ei_pu, cost_grid, f_best_pu],
                      'carbo':[loss_carbo, count_carbo, cost_grid, f_best_carbo],
                         'ei': [loss_ei, count_ei, cost_grid, f_best_ei],
                         'imco': [loss_imco, count_imco, cost_grid, f_best_imco],
                    }


loss_and_count_dict= delete_invalid_loss_and_count(loss_and_count_dict)

'''plot and save results'''
# plot_average_loss(loss_and_count_dict)
plot_and_save_average_loss(loss_and_count_dict, folder, exp_name)

plot_and_save_average_loss_y_best(loss_and_count_dict, folder, exp_name)

'''parameters of method'''
parameter_dict= {'ei_pu':
         {'random_restarts':random_restarts_ei_pu, 'num_iter_max':num_iter_max_ei_pu, 'grid':grid_pu},
                'carbo':
        {'random_restarts':random_restarts_carbo, 'num_iter_max':num_iter_max_carbo, 'grid':grid_carbo},
                 'ei':
         {'random_restarts': random_restarts_ei, 'num_iter_max': num_iter_max_ei, 'grid': grid_ei},
                 'imco':
         {'random_restarts': random_restarts_imco, 'num_iter_max': num_iter_max_imco, 'grid': grid_imco},

                     }
                 # 'pucla_v2':
         # {'num_inner_opt_restarts':num_inner_opt_restarts_v2, 'num_outer_opt_restarts': num_outer_opt_restarts_v2,
         #  'num_iter_max': num_iter_max_pucla_v2, 'grid_opt_in': grid_opt_in_v2, 'grid_opt_out': grid_opt_out_v2}}

'''save results and parameter information'''
save_results(loss_and_count_dict, parameter_dict,folder, exp_name)

