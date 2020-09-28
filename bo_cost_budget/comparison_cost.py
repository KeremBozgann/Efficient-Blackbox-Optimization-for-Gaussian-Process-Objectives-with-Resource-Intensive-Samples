import numpy as np
import gpflow as gp
import GPy
import h5py
from plots import *
from EI_per_cost import EI_per_cost_bo
from EI import EI_bo
from carbo import EI_cool_bo
from ucb_target_lcb_cost import ucb_target_lcb_cost_bo
import matplotlib.pyplot as plt
from EI_pu_with_cost_exploration import EI_pu_with_cost_exploration_bo
import h5py

'''load data'''

dimension= 1
ls= 0.5; var= 1.0; ls_cost= 1.0; var_cost= 1.0; rang= 10; disc= 100; kern_type= 'rbf'

with h5py.File('../datasets/{}d/gp_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(dimension, kern_type, ls, var, rang, disc), 'r') as hf:
    X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

with h5py.File('../datasets/cost_{}d/gp_cost_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(dimension, kern_type, ls_cost, var_cost, rang, disc), 'r') as hf:
    X_cost= np.array(hf.get('X')); Y_cost= np.array(hf.get('y_sample'))

'''plot target and cost function'''
if X.shape[1]==1:
    plot_cost_and_target(X, Y, Y_cost)

elif X.shape[1]==2:
    plot_3D(X[:,0], X[:,1], Y[:,0], title= 'target function')
    plot_3D(X[:,0], X[:,1], Y_cost[:,0], title= 'cost function')

'''set target and cost kernels'''
kernel= gp.kernels.RBF(lengthscales= [ls], variance= var)
kernel_cost= GPy.kern.RBF(input_dim= X.shape[1], lengthscale=  ls_cost, variance= var_cost)

'''set x0, number of iterations and budget'''
budget=10
num_iter= 10
random_index= np.random.permutation(np.arange(X.shape[0]))[0:num_iter]

loss_ei= 0; loss_ei_per_cost=0; cum_cost_ei= 0; cum_cost_ei_per_cost=0

for i in range(num_iter):

    index0= random_index[i]
    loss_list_EI, Xt_EI, Yt_EI, model_EI, cost_list_EI, cum_cost_list_EI= EI_bo(X, Y, Y_cost, kernel, kernel_cost,\
                                       budget, index0, noise= 10**(-4), noise_cost= 10**(-4), plot=False, plot_cost=False)

    # loss_ei+= np.array(loss_list_EI)/num_iter
    # cum_cost_ei+= np.array(cum_cost_list_EI)/num_iter

    loss_list_EI_per_cost, Xt_EI_per_cost, Yt_EI_per_cost, model_EI_per_cost, cost_list_EI_per_cost, cum_cost_list_EI_per_cost\
                        = EI_per_cost_bo(X, Y, Y_cost, kernel, kernel_cost, budget, index0, \
                                 noise= 10**(-4), noise_cost= 10**(-4), plot= False, plot_cost= False)


    # loss_ei_per_cost+= np.array(loss_list_EI_per_cost)/num_iter
    # cum_cost_ei_per_cost += np.array(cum_cost_list_EI_per_cost)/num_iter


    loss_list_carbo, Xt_carbo, Yt_carbo, model_carbo, cost_list_carbo, cum_cost_list_carbo\
                        = EI_cool_bo(X, Y, Y_cost, kernel, kernel_cost, budget, index0, \
                                 noise= 10**(-4), noise_cost= 10**(-4), plot= False, plot_cost= False)

    kapa_targ=3.0; kapa_cost= 1.0

    loss_list_utlc, Xt_utlc, Yt_utlc, model_utlc, cost_list_utlc, cum_cost_list_utlc\
                        =ucb_target_lcb_cost_bo(X, Y, Y_cost, kernel, kernel_cost, budget, index0, \
                                        kapa_targ, kapa_cost, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False)

    kapa_cost= 1.0
    loss_list_ei_pu_cost_exploration, Xt_ei_pu_cost_exploration, Yt_ei_pu_cost_exploration,\
                            model_ei_pu_cost_exploration, cost_list_ei_pu_cost_exploration, cum_cost_list_ei_pu_cost_exploration\
                        =EI_pu_with_cost_exploration_bo(X, Y, Y_cost, kernel, kernel_cost, kapa_cost,
                           budget, index0, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost=False)

    # plt.figure()
    # plt.title('loss vs iteration')
    # plt.plot(np.arange(len(loss_list_EI_per_cost)), np.array(loss_list_EI_per_cost), label= 'ei_per_cost', color= 'blue')
    # plt.scatter(np.arange(len(loss_list_EI_per_cost)), np.array(loss_list_EI_per_cost), label='ei_per_cost', color='blue')
    # plt.plot(np.arange(len(loss_list_EI)), np.array(loss_list_EI), label= 'ei', color= 'red')
    # plt.scatter(np.arange(len(loss_list_EI)), np.array(loss_list_EI), label= 'ei', color= 'red')
    # plt.legend()
    # plt.xlabel('iterations'); plt.ylabel('loss')
    # plt.show()

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

    # input('press a key to continue')
# dict= {'ei':loss_ei, 'ei_per_cost':loss_ei_per_cost}
# cost_dict= {'ei': np.array(cum_cost_list_EI).squeeze(), 'ei_per_cost': np.array(cum_cost_list_EI_per_cost).squeeze()}
#
# comparison_plot_with_dict(dict)
# comparison_cost(cost_dict)

