import numpy as np
import gpflow as gp
import GPy
import h5py

from EI_per_cost import EI_per_cost_bo
from EI import EI_bo
from carbo import EI_cool_bo
from ucb_target_lcb_cost import ucb_target_lcb_cost_bo
import matplotlib.pyplot as plt
from EI_pu_with_cost_exploration import EI_pu_with_cost_exploration_bo
from thompson_sample_cost import thompson_sample_cost_bo

import sys
sys.path.append('../')
from gp_sample_save_to_dataset import *
from plots import *

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

dimension= 2 ;budget=15; num_iter=10

ls= 1.0; var= 1.0; ls_cost= 1.0; var_cost= 1.0; rang= 20; disc= 30; kern_type= 'rbf'; noise= 10**(-4)
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
count_ei_per_cost= np.zeros(cost_grid.shape); loss_ei_per_cost= np.zeros(cost_grid.shape);
count_ei_cool= np.zeros(cost_grid.shape); loss_ei_cool= np.zeros(cost_grid.shape);
count_ei_cost_exploration= np.zeros(cost_grid.shape); loss_ei_cost_exploration= np.zeros(cost_grid.shape);
count_utlc= np.zeros(cost_grid.shape); loss_utlc = np.zeros(cost_grid.shape);
count_cost_thompson= np.zeros(cost_grid.shape); loss_cost_thompson = np.zeros(cost_grid.shape);

for i in range(num_iter):

    if dimension==1:
        X, Y = gp_sample_1d(kern_type, ls, var, rang, disc, noise, save= False, plot= False)
        X_cost, Y_cost= gp_cost_sample_1d(kern_type, ls_cost, var_cost, rang, disc, noise, save= False, plot= False)

    if dimension==2:
        X, Y = gp_sample_2d(kern_type, ls, var, rang, disc, noise, save= False, plot= False)
        X_cost, Y_cost= gp_cost_sample_2d(kern_type, ls_cost, var_cost, rang, disc, noise, save= False, plot= False)

    '''plot target and cost function'''
    if plot_samples==True:
        if X.shape[1]==1:
            plot_cost_and_target(X, Y, Y_cost)

        elif X.shape[1]==2:
            plot_3D(X[:,0], X[:,1], Y[:,0], title= 'target function')
            plot_3D(X[:,0], X[:,1], Y_cost[:,0], title= 'cost function')

    '''set target and cost kernels'''
    kernel= gp.kernels.RBF(lengthscales= [ls], variance= var)
    cost_kernel= gp.kernels.RBF(lengthscales=  [ls_cost], variance= var_cost)

    '''set x0, number of iterations and budget'''

    '''ei'''
    index0= np.random.choice(X.shape[0])
    loss_list_EI, Xt_EI, Yt_EI, model_EI, cost_list_EI, cum_cost_list_EI= EI_bo(X, Y, Y_cost,
                    kernel, budget, index0, cost_kernel, noise=10**(-4), noise_cost= 10**(-4),
                                                            plot=False, plot_cost= False)

    loss_ei, count_ei= add_loss(loss_ei, count_ei, loss_list_EI,   np.atleast_1d(np.array(cum_cost_list_EI).squeeze()),
                                cost_grid)
    # loss_ei+= np.array(loss_list_EI)/num_iter
    # cum_cost_ei+= np.array(cum_cost_list_EI)/num_iter

    '''ei_pu'''
    loss_list_EI_per_cost, Xt_EI_per_cost, Yt_EI_per_cost, model_EI_per_cost, cost_list_EI_per_cost, cum_cost_list_EI_per_cost\
                        = EI_per_cost_bo(X, Y, Y_cost, kernel,
                         cost_kernel, budget, index0, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False)

    loss_ei_per_cost, count_ei_per_cost = add_loss(loss_ei_per_cost, count_ei_per_cost
                                   , loss_list_EI_per_cost, np.atleast_1d(np.array(cum_cost_list_EI_per_cost).squeeze())
                                                   , cost_grid)
    # loss_ei_per_cost+= np.array(loss_list_EI_per_cost)/num_iter
    # cum_cost_ei_per_cost += np.array(cum_cost_list_EI_per_cost)/num_iter

    '''carbo'''
    loss_list_carbo, Xt_carbo, Yt_carbo, model_carbo, cost_list_carbo, cum_cost_list_carbo\
                        = EI_cool_bo(X, Y, Y_cost, kernel,
                     cost_kernel, budget, index0, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False)

    loss_ei_cool, count_ei_cool = add_loss(loss_ei_cool, count_ei_cool, loss_list_carbo,
                                           np.atleast_1d(np.array(cum_cost_list_carbo).squeeze()) , cost_grid)

    '''thompson'''
    kapa_cost= 1.0; num_samp=100

    loss_list_cost_thompson, Xt_cost_thompson, Yt_cost_thompson,\
                            model_cost_thompson, cost_list_cost_thompson, cum_cost_list_cost_thompson\
                        =thompson_sample_cost_bo(X, Y, Y_cost, kernel, cost_kernel, kapa_cost,
                           budget, index0, num_samp, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost=False)

    loss_cost_thompson, count_cost_thompson = add_loss(loss_cost_thompson, count_cost_thompson
                          ,loss_list_cost_thompson,
                      np.atleast_1d(np.array(cum_cost_list_cost_thompson).squeeze()),  cost_grid)

    # kapa_targ=3.0; kapa_cost= 1.0

    # loss_list_utlc, Xt_utlc, Yt_utlc, model_utlc, cost_list_utlc, cum_cost_list_utlc\
    #                     =ucb_target_lcb_cost_bo(X, Y, Y_cost,
    #                 kernel, cost_kernel, budget, index0,  kapa_targ,
    #                         kapa_cost, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False)
    #
    # loss_utlc, count_utlc = add_loss(loss_utlc, count_utlc, loss_list_utlc,
    #                                  np.atleast_1d( np.array(cum_cost_list_utlc).squeeze()),
    #                                     cost_grid)
    #
    # kapa_cost= 1.0
    # loss_list_ei_pu_cost_exploration, Xt_ei_pu_cost_exploration, Yt_ei_pu_cost_exploration,\
    #                         model_ei_pu_cost_exploration, cost_list_ei_pu_cost_exploration, cum_cost_list_ei_pu_cost_exploration\
    #                     =EI_pu_with_cost_exploration_bo(X, Y, Y_cost, kernel, cost_kernel, kapa_cost,
    #                        budget, index0, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost=False)
    #
    # loss_ei_cost_exploration, count_ei_cost_exploration= add_loss(loss_ei_cost_exploration, count_ei_cost_exploration
    #                       , loss_list_ei_pu_cost_exploration,
    #                   np.atleast_1d(np.array(cum_cost_list_ei_pu_cost_exploration).squeeze()),  cost_grid)



    # plt.figure()
    # plt.title('loss vs iteration')
    # plt.plot(np.arange(len(loss_list_EI_per_cost)), np.array(loss_list_EI_per_cost), label= 'ei_per_cost', color= 'blue')
    # plt.scatter(np.arange(len(loss_list_EI_per_cost)), np.array(loss_list_EI_per_cost), label='ei_per_cost', color='blue')
    # plt.plot(np.arange(len(loss_list_EI)), np.array(loss_list_EI), label= 'ei', color= 'red')
    # plt.scatter(np.arange(len(loss_list_EI)), np.array(loss_list_EI), label= 'ei', color= 'red')
    # plt.legend()
    # plt.xlabel('iterations'); plt.ylabel('loss')
    # plt.show()

    if plot_loss_at_each_iteration:

        plt.figure()
        plt.title('loss vs cost')
        plt.plot(np.squeeze(cum_cost_list_EI_per_cost), np.array(loss_list_EI_per_cost), label= 'ei_per_cost', color= 'blue')
        plt.scatter(np.squeeze(cum_cost_list_EI_per_cost), np.array(loss_list_EI_per_cost), label= 'ei_per_cost', color= 'blue')
        plt.plot(np.squeeze(cum_cost_list_EI), np.array(loss_list_EI), label= 'ei', color= 'red')
        plt.scatter(np.squeeze(cum_cost_list_EI), np.array(loss_list_EI), label= 'ei', color= 'red')
        plt.plot(np.squeeze(cum_cost_list_carbo), np.array(loss_list_carbo), label='carbo', color='orange')
        plt.scatter(np.squeeze(cum_cost_list_carbo), np.array(loss_list_carbo), label='carbo', color='orange')

        plt.plot(np.squeeze(cum_cost_list_utlc), np.array(loss_list_utlc), label='utlc', color='green')
        plt.scatter(np.squeeze(cum_cost_list_utlc), np.array(loss_list_utlc), label='utlc', color='green')

        plt.plot(np.squeeze(cum_cost_list_ei_pu_cost_exploration), np.array(loss_list_ei_pu_cost_exploration),\
                                                            label='ei_pu_cost_exploration', color='cyan')
        plt.scatter(np.squeeze(cum_cost_list_ei_pu_cost_exploration), np.array(loss_list_ei_pu_cost_exploration),\
                                                    label='ei_pu_cost_exploration', color='cyan')

        plt.legend()
        plt.xlabel('cost'); plt.ylabel('loss')
        plt.show()



# ei_invalid= np.where(count_ei==0); ei_per_cost_invalid= np.where(count_ei_per_cost==0);
# ei_cool_invalid= np.where(count_ei_cool==0); utlc_invalid= np.where(count_utlc==0);
# ei_cost_exploration_invalid= np.where(count_ei_cost_exploration==0);


loss_and_count_dict= {'ei':[loss_ei, count_ei, cost_grid], 'ei_pu':[loss_ei_per_cost, count_ei_per_cost, cost_grid],
                      'ei_cool':[loss_ei_cool, count_ei_cool,cost_grid],
                      'cost_thompson':[loss_cost_thompson, count_cost_thompson, cost_grid]}


loss_and_count_dict= delete_invalid_loss_and_count(loss_and_count_dict)

# loss_ei_per_cost=  np.delete(loss_ei_per_cost, ei_per_cost_invalid);
# cost_grid_ei_per_cost= np.delete(cost_grid, ei_per_cost_invalid)
#
# loss_ei_cool=  np.delete(loss_ei_cool, ei_cool_invalid);
# cost_grid_ei_cool= np.delete(cost_grid, ei_cool_invalid)
#
# loss_ei_utlc=  np.delete(loss_utlc, utlc_invalid);
# cost_grid_utlc= np.delete(cost_grid, utlc_invalid)
#
# loss_ei_cost_exploration=  np.delete(loss_ei_cost_exploration, ei_cost_exploration_invalid);
# cost_grid_ei_cost_exploration= np.delete(cost_grid, ei_cost_exploration_invalid)
#

# avg_loss_ei= loss_ei/count_ei; avg_loss_ei_pu= loss_ei_per_cost/count_ei_per_cost;
# avg_loss_ei_cool= loss_ei_cool/count_ei_cool; avg_loss_utlc= loss_utlc/count_utlc;
# avg_loss_cost_exploration= loss_ei_cost_exploration/count_ei_cost_exploration;




plot_average_loss(loss_and_count_dict)


    # input('press a key to continue')
# dict= {'ei':loss_ei, 'ei_per_cost':loss_ei_per_cost}
# cost_dict= {'ei': np.array(cum_cost_list_EI).squeeze(), 'ei_per_cost': np.array(cum_cost_list_EI_per_cost).squeeze()}
#
# comparison_plot_with_dict(dict)
# comparison_cost(cost_dict)

