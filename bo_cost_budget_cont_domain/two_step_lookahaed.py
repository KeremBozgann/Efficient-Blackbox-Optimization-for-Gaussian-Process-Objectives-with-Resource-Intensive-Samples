
import numpy as np
import sys
sys.path.append('../')
from gp_gradients import two_opt_EI_optimize

import gpflow as gp
from scipy.stats import norm
import h5py
# sys.path.append('./examples')
# from branin import branin
import sys
import importlib as imp

sys.path.append('../')
import plots
imp.reload(plots)
from plots import *
from gp_gradients import *

sys.path.append('../cost_functions')
sys.path.append('../functions')


from gp_sample_cost_functions import gp_sample_funct


import Acquisitions
imp.reload(Acquisitions)
from Acquisitions import EI_optimize
from Acquisitions import EI

from loss import loss_at_current_step_cont_domain



def two_step_bo(input_dimension, output_dimension, cost_dimension, objective_func, y_true_opt,
                x_true_opt, domain, kernel, evaluation_budget, x0, cost_kernel,
                 num_inner_opt_restarts, num_outer_opt_restarts, monte_carlo_samples, D,
                noise=10 ** (-4), noise_cost=10 ** (-4), plot=False, plot_cost= False):


    Xt = np.zeros([0, input_dimension])
    Yt = np.zeros([0, output_dimension])
    Yt_cost= np.zeros([0,cost_dimension])

    model= gp.models.GPR((Xt,Yt), kernel= kernel)
    model.likelihood.variance.assign(noise)



    # warp_f= GPy.util.warping_functions.LogFunction()
    # warped_cost= GPy.models.WarpedGP(Xt, Yt_cost,kernel_cost, warp_f)
    # warped_cost.likelihood.variance.assign(noise_cost)

    loss_list= []
    cost_list= []
    cum_cost_list= []

    '''Run BO'''

    C= 0
    t=0

    '''grid for the plots'''
    if input_dimension==1:
        disc = 100
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        X= x1.reshape(-1,1)
        Y= objective_func(X)

    if input_dimension==2:
        disc=20
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        x2 = np.linspace(domain[1][0], domain[1][1], disc)
        x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
        X1, X2 = np.meshgrid(x1, x2);

        X1_flat, X2_flat = X1.flatten(), X2.flatten();
        X1_flat, X2_flat = X1.reshape(-1, 1), X2.reshape(-1, 1)
        X = np.append(X1_flat, X2_flat, axis=1)

        Y= objective_func(X)

    f_best_list= []
    while C< evaluation_budget:
        print('t:',t)

        '''Evaluation'''
        if t==0:
            xt= x0; yt= objective_func(xt); yt_cost= gp_sample_funct(xt, cost_kernel, Xt, Yt_cost, noise_cost)
            f_best= yt

        else:

            '''for the plots and acquisition optimization'''
            #
            #
            # Acq_EI = EI(sigma_X, u_X.numpy(), f_best)
            #
            # Acq_dict= {'EI': Acq_EI}


            xt, two_step_improvement_xt, xt_grad, resultt= two_opt_EI_optimize(f_best, Xt, Yt,
                   model, domain, noise, kernel, num_inner_opt_restarts,  num_outer_opt_restarts, monte_carlo_samples, D, Q=1)


            # xt =EI_optimize(model, f_best, domain, X, sigma_X, u_X)

            yt= objective_func(xt); yt_cost= gp_sample_funct(xt, cost_kernel, Xt, Yt_cost, noise_cost)

            if  yt<f_best:
                f_best= yt



        '''cost and target posterior plots'''
        if t > 0:

            if plot == True:
                # if input_dimension==1:
                #     plot_posterior_and_acquisitions_for_continuous_domain(X, Xt, Yt, Yt_cost, xt,
                #                                                           u_X, sigma_X, u_latent_cost,
                #                                                           sigma_latent_cost, Acq_dict, plot_cost, model,
                #                                                           latent_cost_model)

                    # compare_posterior_minimium_approximation_with_grid_search(u_X, X, x_pred_opt, model)

                    # input('press a key to continue')

                if input_dimension==2:

                    plot_target_posterior_cont_domain_2d(X, u_X.numpy(), Xt, Yt, xt, model)

                    # plot_acquisition_for_continuous_domain_2d(X, xt, Acq_dict)

                    # compare_posterior_minimium_approximation_with_grid_search_2d(u_X.numpy(), X, x_pred_opt, model)


                    '''cant implement acquisitions and approximation comparison because of plt.imshow problem'''

                    # plot_acquisitions_cont_domain_2d(x1_max, x2_max, x1_min, x2_min, disc, Acq_dict, xt)

                    input('press a key to continue')

            if plot_cost == True:

                # if X.shape[1] == 1:
                #     cost_posterior_chosen_point_evaluated_points(X, Y_cost, Xt, Yt_cost, xt_cost, yt_cost, u_cost,
                #                                                  sigma_cost)
                #     input('press a key to continue')

                if input_dimension == 2:
                    plot_cost_posterior_cont_domain_2d(X, u_cost, Xt, Yt_cost, xt, latent_cost_model)

                    input('press a key to continue')


        Xt = np.append(Xt, xt, axis=0)
        Yt = np.append(Yt, yt, axis=0)
        Yt_cost= np.append(Yt_cost, yt_cost, axis=0)

        '''update objective model'''
        model = gp.models.GPR((Xt, Yt), kernel=kernel)
        model.likelihood.variance.assign(noise)

        u_X, var_X = model.predict_f(X);
        sigma_X = np.sqrt(var_X)

        '''update cost model'''
        log_Yt_cost= np.log(Yt_cost)
        latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

        u_latent_cost, var_latent_cost = latent_cost_model.predict_f(X);
        sigma_latent_cost = np.sqrt(var_latent_cost)
        u_latent_cost = u_latent_cost.numpy()
        u_cost = np.exp(u_latent_cost)

        loss, x_pred_opt, y_pred_opt= loss_at_current_step_cont_domain(model,
                                                       x_true_opt, y_true_opt, domain, xt, objective_func, X,  u_X )


        loss_list.append(loss)
        cost_list.append(yt_cost)

        '''increment the counter'''

        C+= 1

        print('evaluation number:', C)

        t+= 1
        f_best_list.append(f_best)
        cum_cost_list.append(C)

    if plot==True and input_dimension==2:

        plot_loss_vs_time(loss_list, np.arange(evaluation_budget))
        plot_evaluated_points(Xt, Yt, X, Y)
        plot_posterior(X,Y, model)


    return loss_list, Xt, Yt, model, cost_list, cum_cost_list, f_best_list
#


def test_two_step(input_dimension, output_dimension, cost_dimension, objective_func, y_true_opt,
            x_true_opt, domain, kernel, ls_cost, var_cost, evaluation_budget, x0, noise, noise_cost, D,
                  num_inner_opt_restarts, num_outer_opt_restarts, monte_carlo_samples):


    cost_kernel= gp.kernels.RBF(lengthscales= [ls_cost], variance= var_cost)

    loss_list, Xt, Yt, model, cost_list, cum_cost_list= \
        two_step_bo(input_dimension, output_dimension, cost_dimension, objective_func, y_true_opt,
                    x_true_opt, domain, kernel, evaluation_budget, x0, cost_kernel,
                    num_inner_opt_restarts, num_outer_opt_restarts, monte_carlo_samples, D,
                    noise=10 ** (-4), noise_cost=10 ** (-4), plot=False, plot_cost=False)

    return loss_list, Xt, Yt, model, cost_list, cum_cost_list


sys.path.append('../cost_functions')
sys.path.append('../functions')

'''import sin 1d'''
from sine import sin_opt
from sine import sin_plots
from sine import  sin_find_best_suited_kernel
from sine import sin

'''import camel'''
from six_hump_camel import camel
from six_hump_camel import camel_plots
from six_hump_camel import camel_find_best_suited_kernel

'''import branin'''
from branin import branin
from branin import branin_plots
from branin import branin_find_best_suited_kernel

disc= 20
y_true_opt, x_true_opt, domain= sin_opt()
X, Y= sin_plots(disc)

model ,kernel= sin_find_best_suited_kernel(X, Y, noise=10**(-4))

input_dimension= 1; objective_func= sin;
D= input_dimension
output_dimension= 1; cost_dimension= 1;

x01= np.random.uniform(domain[0][0], domain[0][1]);
# x02= np.random.uniform(domain[1][0], domain[1][1])
x0= np.array([[x01]])

ls_cost= 1.0; var_cost= 1.0
evaluation_budget= 5
noise= 10**(-4); noise_cost= 10**(-4)

num_inner_opt_restarts= 20; num_outer_opt_restarts= 5;  monte_carlo_samples= 10;

loss_list, Xt, Yt, model, cost_list, cum_cost_list= \
    test_two_step(input_dimension, output_dimension, cost_dimension, objective_func, y_true_opt,
            x_true_opt, domain, kernel, ls_cost, var_cost, evaluation_budget, x0, noise, noise_cost, D,
                  num_inner_opt_restarts, num_outer_opt_restarts, monte_carlo_samples)
