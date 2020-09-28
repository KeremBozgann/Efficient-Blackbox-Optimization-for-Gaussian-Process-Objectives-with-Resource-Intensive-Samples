import numpy as np
import gpflow as gp
import GPy
import h5py

import importlib as imp
import Acquisitions
imp.reload(Acquisitions)

from EI_pu_cont_domain import EI_pu_bo_cont_domain
from EI_cont_domain import EI_bo_cont_domain

import sys
sys.path.append('../')
from gp_sample_save_to_dataset import *
from plots import *

sys.path.append('../cost_functions')
sys.path.append('../functions')

from six_hump_camel import camel
from six_hump_camel import camel_plots
from six_hump_camel import camel_find_best_suited_kernel
from six_hump_camel import camel_opt

def find_cell(cost, cost_grid):

    index_list= np.where(cost<=cost_grid)[0]

    if not len(index_list)==0:
        index= index_list[0]
        return index

    else:
        return -1

def add_loss(loss, count, loss_list, cum_cost_list, cost_grid):

    for i in range(len(cum_cost_list)):

        index= find_cell(cum_cost_list[i], cost_grid)

        if not index==-1:
            loss[index]+= loss_list[i]
            count[index]+= 1

    return loss, count


def delete_invalid_loss_and_count(loss_and_count_dict):

    new_dict= {}
    for method in loss_and_count_dict:
        count= loss_and_count_dict[method][1]; loss= loss_and_count_dict[method][0]
        cost_grid= loss_and_count_dict[method][2]

        invalid= np.where(count==0)[0]

        loss= np.delete(loss, invalid)
        count= np.delete(count, invalid)
        cost_grid= np.delete(cost_grid, invalid)
        new_dict[method]= [loss,count,cost_grid ]

    return new_dict

'''load data'''

disc= 21
y_true_opt, x_true_opt, domain= camel_opt()
X, Y= camel_plots(disc)
model ,kernel= camel_find_best_suited_kernel(X, Y, noise=10**(-4))
input_dimension= 2; objective_func= camel;


output_dimension= 1; cost_dimension= 1;


ls_cost= 1.0; var_cost= 1.0
budget= 10
noise= 10**(-4); noise_cost= 10**(-4)

dimension= 2 ;budget=15; num_iter=100

ls= 1.0; var= 1.0; ls_cost= 1.0; var_cost= 1.0; rang= 20; disc= 30; kern_type= 'rbf'; noise= 10**(-4)
cost_kernel= gp.kernels.RBF(lengthscales= [ls_cost], variance= var_cost)

#
# with h5py.File('../datasets/{}d/gp_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(dimension, kern_type, ls, var, rang, disc), 'r') as hf:
#     X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))
#
# with h5py.File('../datasets/cost_{}d/gp_cost_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(dimension, kern_type, ls_cost, var_cost, rang, disc), 'r') as hf:
#     X_cost= np.array(hf.get('X')); Y_cost= np.array(hf.get('y_sample'))


# random_index= np.random.permutation(np.arange(disc**dimension))[0:num_iter]

plot_samples= False;  plot_loss_at_each_iteration= False


cost_grid= np.linspace(0, 3/2*budget, int(3/2*budget*2))


count_ei= np.zeros(cost_grid.shape); loss_ei= np.zeros(cost_grid.shape);
count_ei_pu= np.zeros(cost_grid.shape); loss_ei_pu= np.zeros(cost_grid.shape);


for i in range(num_iter):

    x01 = np.random.uniform(domain[0][0], domain[0][1]);
    x02 = np.random.uniform(domain[1][0], domain[1][1])
    x0 = np.array([[x01, x02]])

    '''ei'''
    loss_list_EI, Xt_EI, Yt_EI, model_EI, cost_list_EI, cum_cost_list_EI= EI_bo_cont_domain(input_dimension, output_dimension,
                        cost_dimension, objective_func, y_true_opt, x_true_opt, domain, kernel, budget, x0,
                                    cost_kernel, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False)

    loss_ei, count_ei= add_loss(loss_ei, count_ei, loss_list_EI,   np.atleast_1d(np.array(cum_cost_list_EI).squeeze()),
                                cost_grid)

    '''ei_pu'''
    loss_list_EI_pu, Xt_EI_pu, Yt_EI_pu, model_EI_pu, cost_list_EI_pu, cum_cost_list_EI_pu\
                        = EI_pu_bo_cont_domain(input_dimension, output_dimension, cost_dimension, objective_func,
                                       y_true_opt, x_true_opt, domain, kernel, budget, x0, cost_kernel, noise=10**(-4),
                                           noise_cost= 10**(-4), plot=False, plot_cost= False)

    loss_ei_pu, count_ei_pu = add_loss(loss_ei_pu, count_ei_pu
                                   , loss_list_EI_pu, np.atleast_1d(np.array(cum_cost_list_EI_pu).squeeze())
                                                   , cost_grid)

    if plot_loss_at_each_iteration:

        plt.figure()
        plt.title('loss vs cost')
        plt.plot(np.squeeze(cum_cost_list_EI_pu), np.array(loss_list_EI_pu), label= 'ei_per_cost', color= 'blue')
        plt.scatter(np.squeeze(cum_cost_list_EI_pu), np.array(loss_list_EI_pu), label= 'ei_per_cost', color= 'blue')
        plt.plot(np.squeeze(cum_cost_list_EI), np.array(loss_list_EI), label= 'ei', color= 'red')
        plt.scatter(np.squeeze(cum_cost_list_EI), np.array(loss_list_EI), label= 'ei', color= 'red')

        plt.legend()
        plt.xlabel('cost'); plt.ylabel('loss')
        plt.show()


loss_and_count_dict= {'ei':[loss_ei, count_ei, cost_grid], 'ei_pu':[loss_ei_pu, count_ei_pu, cost_grid],
                    }


loss_and_count_dict= delete_invalid_loss_and_count(loss_and_count_dict)


plot_average_loss(loss_and_count_dict)


