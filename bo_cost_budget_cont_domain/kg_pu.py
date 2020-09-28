import gpflow as gp
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
from Acquisitions import EI_pu
from Acquisitions import EI

from loss import loss_at_current_step_cont_domain
from kg_pu_opt import Kgpc_optimize

# from ei_pucla_opt import Ei_cepu_optimize
from p_ei_pu_opt import Ei_ppu_optimize


sys.path.append('../HPO')
from keras_model import Keras_model_cifar, Keras_model_fashion


from hyperparameter_optimization import set_and_optimize_gp_model
from hyperparameter_optimization import logistic_bjt

def kgpc_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                 domain, kernel, budget, x0, latent_cost_kernel,num_outer_opt_restarts,  grid_opt_out, grid_opt_in, num_inner_opt_restarts,
              noise=10**(-4),noise_cost= 10**(-4), num_iter_max= 100, plot=False, plot_cost= False, num_layer= None,
              num_dense=None, num_epoch= 1.7, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=False):


    '''constraint values'''
    lower= 10**(-3); upper= 10**(6); #lengtscale and variance constarint
    lower_noise= 10**(-3); upper_noise= 10**(6); #noise constarint

    logistic = logistic_bjt(lower, upper)
    logistic_noise = logistic_bjt(lower_noise, upper_noise)


    '''set initial datapoints and model'''
    if type(X_init)!= np.ndarray:
        Xt = np.zeros([0, D])
        Yt = np.zeros([0, 1])
        Yt_cost= np.zeros([0, 1])

        '''do not optimize models if there is no datapoint'''
        model, latent_cost_model, noise, noise_cost, log_Yt_cost, kernel, latent_cost_kernel= \
                set_and_optimize_gp_model(False, D, Xt, Yt, Yt_cost, noise, noise_cost, kernel, latent_cost_kernel, logistic, logistic_noise)


    else:
        Xt = X_init.copy()
        Yt = Y_init.copy()
        Yt_cost= Y_cost_init.copy()
        f_best= float(np.min(Yt, axis=0))

        if hyper_opt_per!= False and hyper_opt_per>0:
            optimize= True
        else:
            optimize= False

        model, latent_cost_model, noise, noise_cost, log_Yt_cost, kernel, latent_cost_kernel= \
                set_and_optimize_gp_model(optimize, D, Xt, Yt, Yt_cost, noise, noise_cost, kernel, latent_cost_kernel, logistic, logistic_noise)


    loss_list= []
    cost_list= []
    cum_cost_list= []
    '''Run BO'''

    C= 0
    t=0

    '''grid for the plots'''
    if D==1 and (objective_func!= 'cifar' and  objective_func!= 'fashion'):
        disc = 101
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        X= x1.reshape(-1,1)
        Y= objective_func(X)
        Y_cost= cost_function(X)

    if D==2 and (objective_func!= 'cifar' and objective_func!= 'fashion'):
        disc=21
        x1 = np.linspace(domain[0][0], domain[0][1], disc)
        x2 = np.linspace(domain[1][0], domain[1][1], disc)
        x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
        X1, X2 = np.meshgrid(x1, x2);

        X1_flat, X2_flat = X1.flatten(), X2.flatten();
        X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
        X = np.append(X1_flat, X2_flat, axis=1)

        Y= objective_func(X)
        Y_cost= cost_function(X)

    n=20
    opt = Kgpc_optimize(n)

    if Xt.shape[0]!=0:
        '''get u and sigma values for plots'''
        if D==1 or D==2:
            u_X, var_X = model.predict_f(X); u_X= u_X.numpy(); var_X= var_X.numpy()
            sigma_X = np.sqrt(var_X)

            u_latent_cost, var_latent_cost = latent_cost_model.predict_f(X);
            u_latent_cost = u_latent_cost.numpy(); var_latent_cost = var_latent_cost.numpy()
            u_cost = np.exp(u_latent_cost)
            sigma_latent_cost = np.sqrt(var_latent_cost)


    if objective_func=='cifar':
        cifar_model= Keras_model_cifar()
    elif objective_func=='fashion':
        fashion_model= Keras_model_fashion()

    f_best_list = []
    while C<budget:
        print('t:',t)

        '''Evaluation and acquisition optimization'''
        if Xt.shape[0]== 0:
            xt = x0;

            if objective_func == 'fashion':
                layer_sizes = xt[0, 0:num_layer]; alpha = xt[0, num_layer]; l2_regul = xt[0, num_layer + 1];
                # num_epoch = xt[0, num_layer + 2]
                yt, yt_cost, xt = fashion_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul, num_epoch)

            elif objective_func == 'cifar':
                z = num_layer + num_dense; filter_sizes = xt[0, 0:num_layer]; dense_sizes = xt[0, num_layer:z];
                alpha = xt[0, z]; l2_regul = xt[0, z + 1]; dropout = xt[0, z + 2]

                yt, yt_cost, xt = cifar_model.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul, dropout,
                                                                  num_epoch=num_epoch)
            else:
                yt= objective_func(xt); yt_cost= cost_function(xt)

            f_best= yt

        else:

            '''optimize acquisition function on domain'''
            print('entering optimizer')
            xt, xt_val, x1_value_list_grid, X1_grid = \
                opt.maximize_kgpc(kernel, latent_cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
                             num_outer_opt_restarts, num_inner_opt_restarts, grid_opt_out, grid_opt_in, D, log_Yt_cost, num_iter_max, noise_cost)
            print('optimization finished')

            '''evaluation at optimum'''
            if objective_func == 'fashion':
                layer_sizes= xt[0, 0:num_layer]; alpha= xt[0, num_layer]; l2_regul= xt[0,num_layer+1]; #num_epoch= xt[0,num_layer+2]
                yt, yt_cost, xt= fashion_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul, num_epoch)

            elif objective_func == 'cifar':
                z= num_layer+num_dense
                filter_sizes= xt[0, 0:num_layer]; dense_sizes= xt[0,num_layer:z]; alpha= xt[0, z]; l2_regul= xt[0,z+1];
                dropout=xt[0,z+2]
                yt, yt_cost, xt= cifar_model.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul, dropout, num_epoch= num_epoch)

            else:
                yt= objective_func(xt); yt_cost= cost_function(xt)

            if  yt<f_best:
                f_best= yt

            '''get acquisition functions on grid'''
            if grid_opt_out==True and (D==1 or D==2):
                Acq= x1_value_list_grid
                Acq_ei_pu= EI_pu(sigma_X, u_X , f_best, u_cost)
                Acq_ei= EI(sigma_X, u_X , f_best)
                Acq_dict= {'ei_ppu': Acq, 'ei_pu':Acq_ei_pu, 'ei':Acq_ei}

        '''cost and target posterior plots'''
        if t > 0:

            if plot == True:
                if D==1:

                    if grid_opt_out== True:
                        plot_posterior_and_acquisitions_for_continuous_domain(X, Xt, Yt, Yt_cost, xt,
                                                                              u_X, sigma_X, u_latent_cost,
                                                                              sigma_latent_cost, Acq_dict, plot_cost, model,
                                                                              latent_cost_model, Y_cost, Y,
                                                                              plot_true_cost= True, plot_true_targ= True)

                    # compare_posterior_minimium_approximation_with_grid_search(u_X, X, x_pred_opt, model)

                    # input('press a key to continue')

                elif D==2:

                    plot_target_posterior_cont_domain_2d(X, u_X, Xt, Yt, xt, model)

                    # plot_acquisition_for_continuous_domain_2d(X, xt, Acq_dict)

                    # compare_posterior_minimium_approximation_with_grid_search_2d(u_X.numpy(), X, x_pred_opt, model)


                    '''cant implement acquisitions and approximation comparison because of plt.imshow problem'''
                    if grid_opt_out == True:
                        plot_acquisitions_cont_domain_colormap_2d(x1_max, x2_max, x1_min, x2_min, disc, Acq_dict, xt)

                    # input('press a key to continue')

                    if plot_cost == True:

                        plot_cost_posterior_cont_domain_2d(X, u_cost, Xt, Yt_cost, xt, latent_cost_model)
                    #Cost plot for 1d is also plotted above

                    # if X.shape[1] == 1:
                    #     cost_posterior_chosen_point_evaluated_points(X, Y_cost, Xt, Yt_cost, xt_cost, yt_cost, u_cost,
                    #                                                  sigma_cost)
                    #     input('press a key to continue')


                        # input('press a key to continue')

        '''update datapoints'''
        Xt = np.append(Xt, xt, axis=0)
        Yt = np.append(Yt, yt, axis=0)
        Yt_cost= np.append(Yt_cost, yt_cost, axis=0)


        '''update models'''
        if hyper_opt_per != False and hyper_opt_per > 0 and (t + 1) % hyper_opt_per == 0:

            optimize= True
            model, latent_cost_model, noise, noise_cost, log_Yt_cost, kernel, latent_cost_kernel= \
                        set_and_optimize_gp_model(optimize, D, Xt, Yt, Yt_cost, noise, noise_cost, kernel, latent_cost_kernel, logistic, logistic_noise)

        else:
            optimize= False
            model, latent_cost_model, noise, noise_cost, log_Yt_cost, kernel, latent_cost_kernel= \
                        set_and_optimize_gp_model(optimize, D, Xt, Yt, Yt_cost, noise, noise_cost, kernel, latent_cost_kernel, logistic, logistic_noise)


        '''get u and sigma values on grid'''
        if D==1 or D==2:
            u_X, var_X = model.predict_f(X); u_X= u_X.numpy(); var_X= var_X.numpy()
            sigma_X = np.sqrt(var_X)


        if D == 1 or D == 2:
            u_latent_cost, var_latent_cost = latent_cost_model.predict_f(X);
            u_latent_cost = u_latent_cost.numpy(); var_latent_cost = var_latent_cost.numpy()
            u_cost = np.exp(u_latent_cost)
            sigma_latent_cost = np.sqrt(var_latent_cost)


        if objective_func == 'fashion':
            loss, x_pred_opt, y_pred_opt = loss_at_current_step_cont_domain(model, x_true_opt, y_true_opt, domain, xt,
                                              yt, objective_func, None, None, D, random_restarts= 10,
                                         keras_model = fashion_model, num_layer= num_layer, num_dense= None, num_epoch= num_epoch)
        elif objective_func=='cifar':
            loss, x_pred_opt, y_pred_opt= loss_at_current_step_cont_domain(model, x_true_opt, y_true_opt, domain, xt, yt,
                                           objective_func, None,  None, D, random_restarts= 10,
                                         keras_model = cifar_model, num_layer= num_layer, num_dense= num_dense, num_epoch= num_epoch)
        elif D==1 or D==2:
            loss, x_pred_opt, y_pred_opt = loss_at_current_step_cont_domain(model, x_true_opt, y_true_opt, domain, xt, yt,
                                            objective_func, X, u_X, D, random_restarts=10,
                                            keras_model= None, num_layer=None, num_dense=None)
        else:

            loss, x_pred_opt, y_pred_opt = loss_at_current_step_cont_domain(model, x_true_opt, y_true_opt, domain, xt, yt,
                                            objective_func, None, None, D, random_restarts=10,
                                            keras_model= None, num_layer= None, num_dense= None)


        loss_list.append(loss)
        cost_list.append(yt_cost)

        C+= yt_cost
        print('total cost:', C)
        cum_cost_list.append(C.copy())

        t+= 1
        f_best_list.append(f_best)

    if plot==True and D==2:

        plot_loss_vs_time(loss_list, cum_cost_list)
        Y=  objective_func(X)
        plot_evaluated_points(Xt, Yt, X, Y)
        plot_posterior(X,Y, model)


    return loss_list, Xt, Yt, model, cost_list, cum_cost_list, latent_cost_kernel, f_best_list
#

sys.path.append('../cost_functions')
sys.path.append('../functions')

from sine import *
from branin import *
from six_hump_camel import *
from exp_cos_2d import *
from exp_cos_1d import *

def test_synthetic():

    '''dimension dependent assignments'''
    disc= 101
    D=1
    noise = 10 ** (-4); noise_cost = 10 ** (-4);

    objective_func= sin;
    y_true_opt, x_true_opt, domain= sin_opt()
    X, Y= sin_plots(disc, plot= False)
    model ,kernel= sin_find_best_suited_kernel(X, Y, noise=10**(-4))

    X_cost, Y_cost= exp_cos_1d_plots(disc, False)
    latent_cost_model , latent_cost_kernel= exp_cos_1d_find_best_suited_kernel(X_cost, Y_cost, noise=10**(-4))
    cost_function= exp_cos_1d

    lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

    x0= np.random.uniform(lower, upper, (1,D))

    num_outer_opt_restarts=10;
    num_inner_opt_restarts=10;
    num_iter_max=100
    grid_opt_out= False
    grid_opt_in= False
    budget= 20

    loss_list, Xt, Yt, model, cost_list, cum_cost_list, latent_cost_kernel, f_best_list = \
        kgpc_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                domain, kernel, budget, x0, latent_cost_kernel, num_outer_opt_restarts, grid_opt_out, grid_opt_in,
                num_inner_opt_restarts,
                noise= noise, noise_cost=noise_cost, num_iter_max=num_iter_max, plot=False, plot_cost=False, num_layer=None,
                num_dense=None, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=False)

def test_synthetic2():

    '''dimension dependent assignments'''
    disc= 21
    D=2
    noise = 10 ** (-3); noise_cost = 10 ** (-3);

    objective_func= branin;
    y_true_opt, x_true_opt, domain= branin_opt()
    X, Y= branin_plots(disc, plot= False)
    model ,kernel= branin_find_best_suited_kernel(X, Y, noise=10**(-4))

    X_cost, Y_cost= exp_cos_2d_plots(disc, False)
    latent_cost_model , latent_cost_kernel= exp_cos_2d_find_best_suited_kernel(X_cost, Y_cost, noise=10**(-4))
    cost_function= exp_cos_2d

    lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

    x0= np.random.uniform(lower, upper, (1,D))

    num_outer_opt_restarts= 10;
    num_inner_opt_restarts= 10;
    num_iter_max= 100;
    grid_opt_out= False
    grid_opt_in= False

    budget= 20


    loss_list, Xt, Yt, model, cost_list, cum_cost_list, latent_cost_kernel, f_best_list = \
        kgpc_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                domain, kernel, budget, x0, latent_cost_kernel, num_outer_opt_restarts, grid_opt_out, grid_opt_in,
                num_inner_opt_restarts,
                noise= noise, noise_cost=noise_cost, num_iter_max=num_iter_max, plot=False, plot_cost=False, num_layer=None,
                num_dense=None, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=False)




sys.path.append('../HPO')
from keras_model import  get_cifar_domain
from keras_model import  get_fashion_domain
from keras_model import initial_training_cifar
from keras_model import initial_training_fashion
from keras_model import find_best_suited_gp_kernels

from gpflow.utilities import print_summary

def test_cifar():

    sys.path.append('../cost_functions')
    sys.path.append('../functions')


    '''dimension dependent assignments'''
    # disc= 101
    D= 5;

    objective_func= 'cifar'; cost_function= None;
    num_layer= 1; num_dense= 1; num_init_train_samples=5;

    '''initial noise guess for initial hyperparameter optimization'''
    noise= 10**(-3); noise_cost= 10**(-3)
    domain= get_cifar_domain(num_layer, num_dense)

    # y_true_opt, x_true_opt, domain= sin_opt()

    X, Y, Y_cost = initial_training_cifar(domain, num_init_train_samples, num_layer, num_dense)
    kernel, latent_cost_kernel= find_best_suited_gp_kernels(X, Y, Y_cost, noise, noise_cost)

    # X, Y= sin_plots(disc, plot= False)
    # model ,kernel= sin_find_best_suited_kernel(X, Y, noise=10**(-4))

    # X_cost, Y_cost= exp_cos_1d_plots(disc, False)
    # latent_cost_model , latent_cost_kernel= exp_cos_1d_find_best_suited_kernel(X_cost, Y_cost, noise=10**(-4))
    # cost_function= exp_cos_1d

    lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

    x0= np.random.uniform(lower, upper, (1,D))

    budget= 10*60;
    num_outer_opt_restarts= 10;
    num_inner_opt_restarts=10

    grid_opt_out= False
    grid_opt_in= False

    num_iter_max= 100
    y_true_opt= None;  x_true_opt= None

    loss_list, Xt, Yt, model, cost_list, cum_cost_list, latent_cost_kernel, f_best_list = \
        kgpc_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                domain, kernel, budget, x0, latent_cost_kernel, num_outer_opt_restarts, grid_opt_out, grid_opt_in,
                num_inner_opt_restarts,
                noise= noise, noise_cost=noise_cost, num_iter_max=num_iter_max, plot=False, plot_cost=False, num_layer= num_layer,
                num_dense= num_dense, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=False)

def test_fashion():

    sys.path.append('../cost_functions')
    sys.path.append('../functions')


    '''dimension dependent assignments'''
    # disc= 101
    D= 6;


    objective_func= 'fashion'; cost_function= None;
    num_layer= 4;  num_init_train_samples=5;

    '''initial noise guess for initial hyperparameter optimization'''
    noise= 10**(-2); noise_cost= 10**(-2)
    domain= get_fashion_domain(num_layer)

    # y_true_opt, x_true_opt, domain= sin_opt()

    X, Y, Y_cost = initial_training_fashion(domain,  num_init_train_samples, num_layer)
    kernel, latent_cost_kernel= find_best_suited_gp_kernels(X, Y, Y_cost, noise, noise_cost)


    # X, Y= sin_plots(disc, plot= False)
    # model ,kernel= sin_find_best_suited_kernel(X, Y, noise=10**(-4))

    # X_cost, Y_cost= exp_cos_1d_plots(disc, False)
    # latent_cost_model , latent_cost_kernel= exp_cos_1d_find_best_suited_kernel(X_cost, Y_cost, noise=10**(-4))
    # cost_function= exp_cos_1d

    lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

    x0= np.random.uniform(lower, upper, (1,D))


    budget= 60*60;
    num_outer_opt_restarts= 10;
    num_inner_opt_restarts=10

    grid_opt_out= False
    grid_opt_in= False
    num_iter_max = 100
    y_true_opt= None;  x_true_opt= None
    num_dense= None
    loss_list, Xt, Yt, model, cost_list, cum_cost_list, latent_cost_kernel, f_best_list = \
                kgpc_bo(D, objective_func, cost_function, y_true_opt, x_true_opt,
                domain, kernel, budget, x0, latent_cost_kernel, num_outer_opt_restarts, grid_opt_out, grid_opt_in,
                num_inner_opt_restarts,
                noise= noise, noise_cost=noise_cost, num_iter_max=num_iter_max, plot=False, plot_cost=False, num_layer= num_layer,
                num_dense= num_dense, X_init=None, Y_init=None, Y_cost_init=None, hyper_opt_per=False)