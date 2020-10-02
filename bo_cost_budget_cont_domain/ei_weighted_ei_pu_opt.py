import numpy as np
from util import ei_lthc_sampling, ei_sampling
from util import get_mean_and_var_cost, scale
import matplotlib.pyplot as plt
class eiw_eipu():

    def __init__(self):

        pass

    def maximize_eiw_eipu(self, num_lthc_samples, domain, num_ei_samples, model, f_best, latent_cost_model,
                          sampling_method, cut_below_avg):

        if sampling_method== 'lthc':

            X_ei, ei_values= ei_lthc_sampling(num_lthc_samples, num_ei_samples, domain, model, f_best)

        elif sampling_method=='random':
            X_ei, ei_values= ei_sampling(num_lthc_samples, num_ei_samples, domain, model, f_best)

        if cut_below_avg:
            index_above = np.where(ei_values> np.mean(ei_values))[0]
            ei_values_above = ei_values[index_above, :].reshape(-1, 1)
            X_ei_above = X_ei[index_above, :]

            # u_cost_X_ei, _, _= get_mean_and_var_cost(X_ei, latent_cost_model)
            u_cost_X_ei_above, _, _= get_mean_and_var_cost(X_ei_above, latent_cost_model)
            # eipu_values= ei_values/u_cost_X_ei
            eipu_values= ei_values_above/u_cost_X_ei_above


            max_index= np.argmax(eipu_values, axis=0)
            x_opt= X_ei_above[max_index, :].reshape(1,-1)
            value_opt= eipu_values[max_index, :].reshape(1,-1)

        else:
            u_cost_X_ei, _, _= get_mean_and_var_cost(X_ei, latent_cost_model)

            eipu_values = ei_values / u_cost_X_ei

            max_index = np.argmax(eipu_values, axis=0)
            x_opt = X_ei[max_index, :].reshape(1, -1)
            value_opt = eipu_values[max_index, :].reshape(1, -1)

        return x_opt, value_opt

from pyDOE import *
from mpl_toolkits.mplot3d import Axes3D

def test_ei_samples():

    domain = [[-3,3], [-3, 3]]
    D= len(domain)
    X_lthc = lhs(D, samples=1000, criterion='maximin')
    X_lthc= scale(domain, X_lthc)

    values_lthc= (np.exp(np.sin(np.pi*X_lthc[:,0])+ np.sin(np.pi*X_lthc[:,1]))).reshape(-1,1)
    choice= np.random.choice(np.arange(X_lthc.shape[0]), 100, p= values_lthc[:, 0]/np.sum(values_lthc, axis=0))

    X_test= X_lthc[choice , :]
    values_test= values_lthc[choice, :]

    index_above= np.where(values_test >np.mean(values_test))[0]
    values_test_above= values_test[index_above, :].reshape(-1,1)
    X_test_above= X_test[index_above, :]

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter3D(X_lthc [:,0] ,X_lthc [:,1], values_lthc, color= 'red',  alpha= 0.1)
    ax.scatter3D(X_test_above[:, 0], X_test_above[:, 1], values_test_above, color ='blue', alpha= 0.5)
    plt.show()
