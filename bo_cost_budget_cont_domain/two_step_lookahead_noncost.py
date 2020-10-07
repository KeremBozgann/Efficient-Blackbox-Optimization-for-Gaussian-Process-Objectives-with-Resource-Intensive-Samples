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
sys.path.append('../cost_functions')
sys.path.append('../functions')


import Acquisitions
imp.reload(Acquisitions)
from Acquisitions import EI_optimize
from Acquisitions import EI

from loss import loss_at_current_step_cont_domain

from EI_optimize_with_gradient import EI_optimize_with_gradient
from two_step_optimize_with_gradient import two_opt_EI_optimize_herm

def two_step_bo_cont_domain_nocost(input_dimension, output_dimension, objective_func, y_true_opt,
                      x_true_opt, domain, kernel, evaluation_budget, x0,  num_outer_opt_restarts,
                       num_inner_opt_restarts, grid_opt_in, grid_opt_out, noise=10**(-4), random_restarts=10, num_iter_max= 100, plot=False):


    '''Find true optimal point'''

    D= input_dimension

    Xt = np.zeros([0, input_dimension])
    Yt = np.zeros([0, output_dimension])

    model= gp.models.GPR((Xt,Yt), kernel= kernel)
    model.likelihood.variance.assign(noise)

    loss_list= []

    '''Run BO'''

    t=0

    '''grid for the plots'''
    if input_dimension==1:
        disc = 101
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        X= x1.reshape(-1,1)
        Y= objective_func(X)

    if input_dimension==2:
        disc=21
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        x2 = np.linspace(domain[1][0], domain[1][1], disc)
        x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
        X1, X2 = np.meshgrid(x1, x2);

        X1_flat, X2_flat = X1.flatten(), X2.flatten();
        X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
        X = np.append(X1_flat, X2_flat, axis=1)

        Y= objective_func(X)

    while t<evaluation_budget:
        print('t:',t)

        '''Evaluation'''
        if t==0:
            xt= x0; yt= objective_func(xt);
            f_best= yt

        else:

            '''for the plots and acquisition optimization'''
            xt, xt_val, x1_value_list_grid, X1_grid = \
                                        two_opt_EI_optimize_herm(f_best, Xt, Yt, model, domain, noise, kernel,
                                                  num_inner_opt_restarts, num_outer_opt_restarts, D, num_iter_max,
                                                        grid_opt_in, grid_opt_out)


            #assuming that X1_grid= XAcq = x1_value_list_grid

            Acq_EI= EI(sigma_X, u_X, f_best)

            if grid_opt_out==True:
                Acq_dict = {'TS':x1_value_list_grid, 'EI': Acq_EI}

            yt= objective_func(xt);

            if  yt<f_best:
                f_best= yt


        '''target posterior plots'''
        if t > 0:

            if plot == True:
                if input_dimension==1:

                    if grid_opt_out==True:
                        plot_posterior_and_acquisitions_for_continuous_domain_noncost(X, Xt, Yt, xt,
                                                                                      u_X, sigma_X, Acq_dict, model)

                    compare_posterior_minimium_approximation_with_grid_search(u_X, X, x_pred_opt, model)

                    # input('press a key to continue')

                elif input_dimension==2:

                    # plot_target_posterior_cont_domain_2d(X, u_X, Xt, Yt, xt, model)

                    # plot_acquisition_for_continuous_domain_2d(X, xt, Acq_dict)

                    # compare_posterior_minimium_approximation_with_grid_search_2d(u_X.numpy(), X, x_pred_opt, model)


                    '''cant implement acquisitions and approximation comparison because of plt.imshow problem'''

                    if grid_opt_out== True:
                        plot_acquisitions_cont_domain_colormap_2d(x1_max, x2_max, x1_min, x2_min, disc, Acq_dict, xt)

                    # input('press a key to continue')


        Xt = np.append(Xt, xt, axis=0)
        Yt = np.append(Yt, yt, axis=0)

        '''update objective model'''
        model = gp.models.GPR((Xt, Yt), kernel=kernel)
        model.likelihood.variance.assign(noise)

        u_X, var_X = model.predict_f(X); u_X= u_X.numpy(); var_X= var_X.numpy()
        sigma_X = np.sqrt(var_X)


        loss, x_pred_opt, y_pred_opt= loss_at_current_step_cont_domain(model,
                                       x_true_opt, y_true_opt, domain, xt, objective_func, X,  u_X, D, random_restarts )


        loss_list.append(loss)
        '''increment the counter'''

        t+= 1
        print('t',t)

    if plot==True and input_dimension==2:

        plot_loss_vs_evaluation(loss_list, evaluation_budget)
        Y=  objective_func(X)
        plot_evaluated_points(Xt, Yt, X, Y)
        plot_posterior(X,Y, model)


    return loss_list, Xt, Yt, model
#


def test_two_step(input_dimension, output_dimension, objective_func, y_true_opt,
            num_outer_opt_restarts, random_restarts, num_inner_opt_restarts, x_true_opt, domain, kernel,
                     evaluation_budget, x0, noise, plot, num_iter_max, grid_opt_in, grid_opt_out):


    loss_list, Xt, Yt, model= \
        two_step_bo_cont_domain_nocost(input_dimension, output_dimension, objective_func, y_true_opt,
                           x_true_opt, domain, kernel, evaluation_budget, x0, num_outer_opt_restarts,
                               num_inner_opt_restarts, grid_opt_in, grid_opt_out, noise=noise, random_restarts=random_restarts, num_iter_max=num_iter_max,
                                   plot=plot)

    return loss_list, Xt, Yt, model

#
# sys.path.append('../cost_functions')
# sys.path.append('../functions')
# from sine import *
# from branin import *
# from six_hump_camel import camel
# from six_hump_camel import camel_plots
# from six_hump_camel import camel_find_best_suited_kernel
# from six_hump_camel import camel_opt
#
# disc= 21
# y_true_opt, x_true_opt, domain= branin_opt()
# X, Y= branin_plots(disc)
#
# model ,kernel= branin_find_best_suited_kernel(X, Y, noise=10**(-4))
# input_dimension= 2; objective_func= branin;
# D= input_dimension
#
# lower = [domain[i][0] for i in range(len(domain))];
# upper = [domain[i][1] for i in range(len(domain))]
#
# x0=  np.random.uniform(lower, upper, (1,D))
#
# output_dimension= 1;
#
# evaluation_budget=5
# noise= 10**(-4)
# num_iter_max= 100
#
# plot= True
#
# num_inner_opt_restarts= 0
# num_outer_opt_restarts= 0
#
# random_restarts= 20
# grid_opt_in= True
# grid_opt_out= True
#
# loss_list, Xt, Yt, model= test_two_step(input_dimension, output_dimension, objective_func, y_true_opt,
#             num_outer_opt_restarts, random_restarts, num_inner_opt_restarts, x_true_opt, domain, kernel,
#                      evaluation_budget, x0, noise, plot, num_iter_max, grid_opt_in, grid_opt_out)

