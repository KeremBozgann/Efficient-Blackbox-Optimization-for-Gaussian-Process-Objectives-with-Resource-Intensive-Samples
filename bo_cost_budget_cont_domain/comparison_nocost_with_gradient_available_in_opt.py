import numpy as np
import gpflow as gp
import h5py

import importlib as imp
import Acquisitions
imp.reload(Acquisitions)

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

from sine import sin
from sine import sin_plots
from sine import sin_find_best_suited_kernel
from sine import sin_opt

from branin import branin
from branin import branin_plots
from branin import branin_find_best_suited_kernel
from branin import branin_opt


from EI_cont_domain_nocost import EI_bo_cont_domain_nocost
from two_step_lookahead_noncost import  two_step_bo_cont_domain_nocost


disc= 21
y_true_opt, x_true_opt, domain= branin_opt()
X, Y= branin_plots(disc)
model ,kernel= branin_find_best_suited_kernel(X, Y, noise=10**(-4))
input_dimension=2; objective_func= branin;
output_dimension= 1;
D= input_dimension

noise= 10**(-4);
num_iter=20

plot_samples= False;  plot_loss_at_each_iteration= False

evaluation_budget= 20

'''ei parameters'''
grid_ei= True; plot_ei= False
random_restarts_ei= 20
opt_restarts_ei= 0
'''two step parameters'''
num_outer_opt_restarts = 0;
num_inner_opt_restarts = 0
grid_opt_in = True
grid_opt_out = True
random_restarts_ts = 20
plot_ts= False

loss_list_ei_list= np.zeros([0, evaluation_budget])
loss_list_ts_list= np.zeros([0, evaluation_budget])

print('entering for loop')
for i in range(num_iter):

    print('iteration:{} begins'.format(i))
    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]
    x0= np.random.uniform(lower, upper, (1,D))


    '''ei'''
    loss_list_ei, Xt_ei, Yt_ei, model_ei = \
        EI_bo_cont_domain_nocost(input_dimension, output_dimension, objective_func, y_true_opt,
                      x_true_opt, domain, kernel, evaluation_budget, x0, grid_ei, noise=noise,
                             opt_restarts= opt_restarts_ei, random_restarts=random_restarts_ei, num_iter_max= 100, plot=plot_ei)
    loss_list_ei= np.array(loss_list_ei); loss_list_ei= loss_list_ei.reshape(1,-1)
    loss_list_ei_list= np.append(loss_list_ei_list, np.array(loss_list_ei), axis= 0)


    '''ei_two_step'''
    loss_list_ts, Xt_ts, Yt_ts, model_ts=  \
        two_step_bo_cont_domain_nocost(input_dimension, output_dimension, objective_func, y_true_opt,
                                       x_true_opt, domain, kernel, evaluation_budget, x0, num_outer_opt_restarts,
                                       num_inner_opt_restarts, grid_opt_in, grid_opt_out, noise= noise,
                                       random_restarts=random_restarts_ts, num_iter_max=100, plot=plot_ts)
    loss_list_ts= np.array(loss_list_ts); loss_list_ts= loss_list_ts.reshape(1,-1)
    loss_list_ts_list = np.append(loss_list_ts_list, loss_list_ts, axis=0)



loss_list_ei_avg = np.mean(loss_list_ei_list, axis=0)
loss_list_ts_avg = np.mean(loss_list_ts_list, axis=0)
plt.figure()
plt.plot( np.arange(evaluation_budget), loss_list_ei_avg, color= 'red', label= 'ei')
plt.scatter( np.arange(evaluation_budget), loss_list_ei_avg ,color= 'red', label= 'ei')

plt.plot( np.arange(evaluation_budget), loss_list_ts_avg, color= 'blue', label= 'ts')
plt.scatter( np.arange(evaluation_budget), loss_list_ts_avg, color= 'blue', label= 'ts')

plt.legend()
plt.xlabel('evaluation')
plt.ylabel('loss')
plt.show()