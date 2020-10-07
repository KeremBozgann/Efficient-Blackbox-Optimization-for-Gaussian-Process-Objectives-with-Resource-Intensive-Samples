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


# from gp_sample_cost_functions import gp_sample_funct


import Acquisitions
imp.reload(Acquisitions)
from Acquisitions import EI_pu_optimize
from Acquisitions import EI_pu
from Acquisitions import EI

from loss import loss_at_current_step_cont_domain

from ei_pucla_opt_v2 import Ei_cepu_v2_optimize


def ei_pucla_v2_bo(input_dimension, output_dimension, cost_dimension, objective_func, cost_function, y_true_opt, x_true_opt,
                 domain, kernel, budget, x0, cost_kernel,num_outer_opt_restarts, num_inner_opt_restarts,
                        grid_opt_in, grid_opt_out, noise=10**(-4), noise_cost= 10**(-4), num_iter_max= 100,
                        plot=False, plot_cost= False):

    D = input_dimension

    '''Find true optimal point'''

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
        disc = 101
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        X= x1.reshape(-1,1)
        Y= objective_func(X)
        Y_cost= cost_function(X)

    if input_dimension==2:
        disc=21
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        x2 = np.linspace(domain[1][0], domain[1][1], disc)
        x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
        X1, X2 = np.meshgrid(x1, x2);

        X1_flat, X2_flat = X1.flatten(), X2.flatten();
        X1_flat, X2_flat = X1.reshape(-1, 1), X2.reshape(-1, 1)
        X = np.append(X1_flat, X2_flat, axis=1)

        Y= objective_func(X)
        Y_cost= cost_function(X)

    opt =Ei_cepu_v2_optimize()

    f_best_list= []
    while C<budget:
        print('t:',t)

        '''Evaluation'''
        if t==0:
            xt= x0; yt= objective_func(xt); yt_cost= cost_function(xt)
            f_best= yt

        else:

            '''for the plots and acquisition optimization'''

            print('entering optimizer')
            xt, xt_val, x1_value_list_grid, X1_grid = \
                opt.maximize_ei_cepu(kernel, cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
                             num_inner_opt_restarts, num_outer_opt_restarts, grid_opt_in, grid_opt_out, D, log_Yt_cost, num_iter_max, noise_cost)
            print('optimization finished')
            yt= objective_func(xt); yt_cost= cost_function(xt)

            if grid_opt_out==True:
                Acq= x1_value_list_grid
                Acq_ei_pu= EI_pu(sigma_X, u_X , f_best, u_cost)
                Acq_ei= EI(sigma_X, u_X , f_best)
                Acq_dict= {'ei_pucla': Acq, 'ei_pu':Acq_ei_pu,'ei':Acq_ei}

            if  yt<f_best:
                f_best= yt



        '''cost and target posterior plots'''
        if t > 0:

            if plot == True:
                if input_dimension==1:

                    if grid_opt_out== True:
                        plot_posterior_and_acquisitions_for_continuous_domain(X, Xt, Yt, Yt_cost, xt,
                                                                              u_X, sigma_X, u_latent_cost,
                                                                              sigma_latent_cost, Acq_dict, plot_cost, model,
                                                                              latent_cost_model, Y_cost, plot_true_cost= True)

                    # compare_posterior_minimium_approximation_with_grid_search(u_X, X, x_pred_opt, model)

                    input('press a key to continue')

                elif input_dimension==2:

                    plot_target_posterior_cont_domain_2d(X, u_X, Xt, Yt, xt, model)

                    # plot_acquisition_for_continuous_domain_2d(X, xt, Acq_dict)

                    # compare_posterior_minimium_approximation_with_grid_search_2d(u_X.numpy(), X, x_pred_opt, model)


                    '''cant implement acquisitions and approximation comparison because of plt.imshow problem'''

                    if grid_opt_out== True:
                        plot_acquisitions_cont_domain_colormap_2d(x1_max, x2_max, x1_min, x2_min, disc, Acq_dict, xt)

                    input('press a key to continue')

            if plot_cost == True:

                #Cost plot for 1d is also plotted above

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

        u_X, var_X = model.predict_f(X); u_X= u_X.numpy(); var_X= var_X.numpy()
        sigma_X = np.sqrt(var_X)


        '''update cost model'''
        log_Yt_cost= np.log(Yt_cost)
        latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

        u_latent_cost, var_latent_cost = latent_cost_model.predict_f(X);
        u_latent_cost = u_latent_cost.numpy(); var_latent_cost = var_latent_cost.numpy()
        u_cost = np.exp(u_latent_cost)
        sigma_latent_cost = np.sqrt(var_latent_cost)

        loss, x_pred_opt, y_pred_opt= loss_at_current_step_cont_domain(model,
                                       x_true_opt, y_true_opt, domain, xt, objective_func, X,  u_X, D, random_restarts= 10 )


        loss_list.append(loss)
        cost_list.append(yt_cost)
        '''increment the counter'''

        C+= yt_cost
        print('total cost:', C)
        cum_cost_list.append(C.copy())

        t+= 1
        f_best_list.append(f_best)

    if plot==True and input_dimension==2:

        plot_loss_vs_time(loss_list, cum_cost_list)
        Y=  objective_func(X)
        plot_evaluated_points(Xt, Yt, X, Y)
        plot_posterior(X,Y, model)


    return loss_list, Xt, Yt, model, cost_list, cum_cost_list, f_best_list
#


def test_ei_pucla_v2(input_dimension, output_dimension, cost_dimension, objective_func, cost_function, y_true_opt,
            x_true_opt, domain, kernel, latent_cost_kernel, budget, x0, noise, noise_cost,
              num_outer_opt_restarts, num_inner_opt_restarts, grid_opt_in, grid_opt_out):


    loss_list, Xt, Yt, model, cost_list, cum_cost_list= ei_pucla_v2_bo(input_dimension, output_dimension, cost_dimension,
            objective_func, cost_function, y_true_opt, x_true_opt, domain, kernel, budget, x0, latent_cost_kernel,num_outer_opt_restarts, num_inner_opt_restarts,
                    grid_opt_in, grid_opt_out, noise=noise_cost, noise_cost=noise, num_iter_max=100, plot=True, plot_cost= True)

    return loss_list, Xt, Yt, model, cost_list, cum_cost_list


sys.path.append('../cost_functions')
sys.path.append('../functions')

from sine import *
from branin import *
from six_hump_camel import *
from exp_cos_2d import *
from exp_cos_1d import *

'''dimension dependent assignments'''
disc= 101
input_dimension= 1;

objective_func= sin;
y_true_opt, x_true_opt, domain= sin_opt()
X, Y= sin_plots(disc, plot= False)
model ,kernel= sin_find_best_suited_kernel(X, Y, noise=10**(-4))

X_cost, Y_cost= exp_cos_1d_plots(disc, False)
latent_cost_model , latent_cost_kernel= exp_cos_1d_find_best_suited_kernel(X_cost, Y_cost, noise=10**(-4))
cost_function= exp_cos_1d



D= input_dimension

lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

x0= np.random.uniform(lower, upper, (1,D))

output_dimension= 1; cost_dimension= 1;

budget= 10
noise= 10**(-4); noise_cost= 10**(-4)

num_outer_opt_restarts= 0; num_inner_opt_restarts= 0
grid_opt_in= True; grid_opt_out= True

loss_list, Xt, Yt, model, cost_list, cum_cost_list = test_ei_pucla_v2(input_dimension, output_dimension, cost_dimension, objective_func,
                    cost_function, y_true_opt, x_true_opt, domain, kernel, latent_cost_kernel, budget, x0, noise, noise_cost,
                            num_outer_opt_restarts, num_inner_opt_restarts,grid_opt_in, grid_opt_out)




