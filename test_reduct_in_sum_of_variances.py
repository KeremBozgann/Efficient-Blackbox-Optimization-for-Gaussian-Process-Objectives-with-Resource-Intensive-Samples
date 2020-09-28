import numpy as np
import matplotlib.pyplot as plt
import gpflow as gp
import h5py
import tensorflow as tf
from plots import *

noise= 10**(-4)

with h5py.File('./datasets/1d/gp_sample_rbf_l_1.0_v_1.0.h5', 'r') as hf:
    X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

ind_rand= np.random.permutation(np.arange(X.shape[0]))[0:1]
Xt= X[ind_rand,:]; Yt= Y[ind_rand, :]
kernel= gp.kernels.RBF()
model= gp.models.GPR((Xt,Yt), kernel= kernel)
model.likelihood.variance.assign(noise)





'''posterior'''
mu, var= model.predict_f(X)
sum_of_stds= np.sum(np.sqrt(var),axis=0)

'''posterior with pseudo evaluations'''
list_reduction_sum_of_variances= []

for i in range(X.shape[0]):
    x_psed = X[i, :].reshape(1, -1); Xt_psed = np.append(Xt, x_psed, axis=0)

    '''dummy observation'''
    y_dum= np.random.rand(1, 1); Yt_psed = np.append(Yt, y_dum, axis=0)

    '''pseudo model'''
    model_pse_i= gp.models.GPR((Xt_psed,Yt_psed), kernel= kernel)
    model_pse_i.likelihood.variance.assign(noise)

    mu_psed, var_psed= model_pse_i.predict_f(X)

    sum_of_pseudo_stds= np.sum(np.sqrt(var_psed),axis=0)

    reduct_sum_of_stds= sum_of_stds- sum_of_pseudo_stds
    list_reduction_sum_of_variances.append(reduct_sum_of_stds)

index_chosen= np.argmax(list_reduction_sum_of_variances)
x_chosen= X[index_chosen,:].reshape(-1,1); y_chosen= Y[index_chosen,:].reshape(-1,1)

'''show the point that has results in maximum entropy reduction'''
scatter_chosen_point_and_previously_evaluated_points_and_plot_posterior(X, Xt, Yt, x_chosen, y_chosen, mu, np.sqrt(var))