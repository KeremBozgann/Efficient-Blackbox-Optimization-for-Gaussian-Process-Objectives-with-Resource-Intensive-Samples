

import numpy as np
import GPy
from scipy.spatial import distance
import gpflow as gp
from scipy.stats import norm

import h5py
import sys
sys.path.append('../')
from plots import *

def update_cost_model(X_init, Y_cost_init, cost_kernel, noise_cost):

    if X_init.shape[0]> 0:
        log_Y_cost_init= np.log(Y_cost_init)
        latent_cost_model= gp.models.GPR((X_init,log_Y_cost_init), cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

        return latent_cost_model

    else:
        return None

def get_distance_to_X_init(temp_X, X_init):

    dist_matrix= distance.cdist(temp_X, X_init)
    dist= np.min(dist_matrix, axis=1)

    return dist

def get_chosen_index(X_cand, x_chosen):

    row_index= np.where((X_cand == x_chosen).all(axis=1))[0]

    return row_index

def initial_evaluations(X, Y, Y_cost, cost_kernel, budget, budget_init, index0, noise_cost):


    X_init = X[index0, :].reshape(1,-1); Y_cost_init= Y_cost[index0, :].reshape(1,-1); Y_init= Y[index0, :].reshape(1,-1)

    X_cand= np.delete(X, index0, axis=0); Y_cost_cand= np.delete(Y_cost, index0, axis=0); Y_cand= np.delete(Y, index0, axis=0)

    c= Y_cost_init[0,0]

    t=0
    while c<budget_init :

        if t>0:
            X_init = np.append(X_init, x_chosen, axis= 0)
            Y_cost_init= np.append(Y_cost_init, y_cost_chosen, axis=0)
            Y_init = np.append(Y_init, y_chosen, axis=0)

            index_chosen= get_chosen_index(X_cand, x_chosen)
            X_cand= np.delete(X_cand, index_chosen, axis=0)
            Y_cost_cand= np.delete(Y_cost_cand, index_chosen, axis=0)
            Y_cand = np.delete(Y_cand, index_chosen, axis=0)

        if X_cand.shape[0]==1:
            x_chosen = X_cand.copy();
            y_cost_chosen = Y_cost_cand.copy();
            y_chosen =Y_cand.copy()

            X_init= np.append(X_init, x_chosen, axis=0);
            Y_cost_init= np.append(Y_cost_init, y_cost_chosen, axis=0)
            Y_init= np.append(Y_init, y_chosen, axis=0)

            c+= y_cost_chosen

            break

        latent_cost_model= update_cost_model(X_init, Y_cost_init, cost_kernel, noise_cost)
        u_latent_cost, var_latent_cost = latent_cost_model.predict_f(X_cand); u_cost= np.exp(u_latent_cost)

        temp_X= X_cand.copy(); temp_Y_cost= Y_cost_cand.copy(); temp_Y= Y_cand.copy()

        while temp_X.shape[0]>1:

            '''exclude most cosly point'''
            index_max_cost= np.argmax(u_cost)
            temp_X= np.delete(temp_X, index_max_cost, axis=0); temp_Y_cost= np.delete(temp_Y_cost, index_max_cost, axis=0)
            temp_Y= np.delete(temp_Y, index_max_cost, axis=0)

            u_cost= np.delete(u_cost, index_max_cost, axis=0)

            if temp_X.shape[0]==1:
                break
            '''exclude closest point to X_init '''
            index_max_dist= np.argmax(get_distance_to_X_init(temp_X, X_init))
            temp_X = np.delete(temp_X, index_max_dist, axis=0); temp_Y_cost= np.delete(temp_Y_cost, index_max_dist, axis=0)
            temp_Y= np.delete(temp_Y, index_max_dist, axis=0)

            u_cost = np.delete(u_cost, index_max_dist, axis=0);

        x_chosen= temp_X.copy(); y_cost_chosen= temp_Y_cost.copy(); y_chosen= temp_Y.copy()
        c+= y_cost_chosen
        t+=1

    return X_init, Y_cost_init, Y_init, c

def EI_per_cost(sigma_x, u_x , f_best, u_cost):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))

    EI_per_cost= EI_x/u_cost

    return EI_per_cost

def EI(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))


    return EI_x

def thompson_sample_cost(sigma_X, u_X, f_best, X, Xt, Yt_cost, cost_kernel, noise_cost, num_samp):

    ei= EI(sigma_X, u_X, f_best)

    loss_total= np.zeros([u_X.shape[0],1])

    log_Yt_cost= np.log(Yt_cost)

    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), cost_kernel )
    latent_cost_model.likelihood.variance.assign(noise_cost)


    # for i in range(num_samp):
    #
    #     latent_sample= latent_cost_model.predict_f_samples(X)
    #     cost_sample=  np.exp(latent_sample)
    #
    #     ei_pu_s = ei/cost_sample;  ei_pu_maks= np.amax(ei_pu_s, axis=0)
    #     loss_s= ei_pu_maks- ei_pu_s;
    #     loss_total+= loss_s
    #
    # acq= -loss_total

    ei_pu_total= np.zeros([u_X.shape[0],1])

    for i in range(num_samp):

        latent_sample= latent_cost_model.predict_f_samples(X)
        cost_sample=  np.exp(latent_sample)

        ei_pu_s = ei/cost_sample
        ei_pu_total+= ei_pu_s


    return ei_pu_total

def EI_cool(sigma_x, u_x , f_best, u_cost, C, budget, budget_init):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))

    EI_cool= EI_x/(u_cost)**((budget- C)/(budget- budget_init))

    return EI_cool


def EI_pu_with_cost_exploration(u_X, sigma_X, u_latent_cost, sigma_latent_cost,
                                               f_best, cost_kernel, noise_cost, Xt, Yt_cost, X, kapa_cost):

    ei= EI(sigma_X, u_X, f_best)
    ei_pu= ei/np.exp(u_latent_cost)

    # EI_pu_with_additional_cost_acquisition= np.empty([X.shape[0],1])

    # for i in range(X.shape[0]):
    #
    #     '''pseudo evaluations'''
    #     x_pse = X[i, :].reshape(1, -1); y_dum = np.random.rand(1, 1);
    #     Xt_pse = np.append(Xt, x_pse, axis=0); Yt_pse = np.append(Yt_cost, y_dum, axis=0)
    #
    #     '''pseudo model'''
    #     cost_model_pse = GPy.models.WarpedGP(Xt_pse, Yt_pse, kernel_cost, warp_f)
    #     cost_model_pse.likelihood.variance= noise_cost
    #
    #     u_pse_cost, var_pse_cost= cost_model_pse.predict(X); sigma_pse_cost= np.sqrt(var_pse_cost)
    #
    #     EI_pu_upper = ei / (np.clip(u_cost - kapa_cost * sigma_pse_cost, a_min=10 ** (-3), a_max=np.inf))
    #     EI_pu_lower = ei / (np.clip(u_cost + kapa_cost * sigma_pse_cost, a_min=10 ** (-3), a_max=np.inf))
    #
    #     EI_pu_upper= np.delete(EI_pu_upper, i, axis=0); EI_pu_lower= np.delete(EI_pu_lower, i, axis=0)
    #     ei_pu_temp= np.delete(ei_pu, i, axis=0)
    #
    #     max_EI_pu_i= np.amax(EI_pu_upper, axis=0)
    #
    #     cost_additional_i= np.sum(np.clip((max_EI_pu_i- EI_pu_lower),a_min= 10**(-3), a_max= np.inf)\
    #                                     *ei_pu_temp/np.sum(ei_pu_temp, axis=0), axis=0)
    #
    #
    #
    #     EI_pu_with_additional_cost_acquisition[i,0]= ei_pu[i,0]- cost_additional_i
    ei_pu_with_exploration_term = ei_pu+ 2*kapa_cost*sigma_latent_cost*ei_pu

    return ei_pu_with_exploration_term


def ucb_target_lcb_cost(u_X, sigma_X, u_latent_cost,sigma_latent_cost, kapa_targ, kapa_cost, f_best):


    utlc= np.clip(f_best- (u_X- kapa_targ*sigma_X), a_min= 0, a_max= np.inf)/ \
          np.clip(np.exp(u_latent_cost- kapa_cost*sigma_latent_cost), a_min= 10**(-3), a_max= np.inf)


    return utlc

def one_step_lookahead(sigma_X, u_X, u_cost, f_best, num_samp, X,  Xt, Yt, model, kernel, noise):

    ei0= EI(sigma_X, u_X , f_best)

    acq= np.zeros([X.shape[0], 1])

    for i in range(X.shape[0]):

        ei0_x = ei0[i, 0]

        total_ei= 0

        x = X[i, :].reshape(1, -1)
        Xt_psei = np.append(Xt, x, axis=0)

        for j in range(num_samp):

            y_pseij= model.predict_f_samples(x);

            Yt_pseij= np.append(Yt, y_pseij, axis=0);
            model_pseij= gp.models.GPR((Xt_psei, Yt_pseij),kernel)
            model_pseij.likelihood.variance.assign(noise)

            if y_pseij<f_best:
                f_best_pseij= y_pseij
            else:
                f_best_pseij= f_best.copy()

            u_pseij, var_pseij= model_pseij.predict_f(X); sigma_pseij= np.sqrt(var_pseij)

            ei1ij= EI(sigma_pseij, u_pseij, f_best_pseij)

            index_x2= int(np.argmax(ei1ij/u_cost, axis=0)); ei1ij_x2= ei1ij[index_x2, 0]


            total_ei+= (ei0_x+ ei1ij_x2)

        avg_total_ei= total_ei/num_samp

        acq[i, 0]= avg_total_ei/(u_cost[i,0]+ u_cost[index_x2, 0])

    return acq


def loss_at_current_step(X, Y, model, x_true_opt, y_true_opt):
    '''Get posterior mean and variance'''
    # model= gp.models.GPR((Xt,Yt),kernel= kern_temp)
    u_X, sigma_X = model.predict_f(X)

    '''Find estimated optimal point from posterior'''
    index_pred_opt = np.argmin(u_X, axis=0)
    x_pred_opt = X[index_pred_opt, :]
    y_pred_opt = Y[index_pred_opt, :]

    '''Determine the squared loss for the prediction'''
    print('optimum predicted design point:{}, objective value at predicted optimum:{}'.format(x_pred_opt, y_pred_opt))
    print('true optimum design point:{}, objective value at optimum:{}'.format(x_true_opt, y_true_opt))
    loss = np.square(float(y_pred_opt - y_true_opt))
    print('loss of predicted point:', loss)

    return loss, index_pred_opt, x_pred_opt, y_pred_opt



def one_step_lookahead_bo(X, Y, Y_cost, kernel, cost_kernel, kapa_cost,
                           budget, index0, num_samp, noise=10**(-4), noise_cost= 10**(-4), plot=True, plot_cost=True):

    '''Find true optimal point'''
    index_true_opt= np.argmin(Y,axis=0)
    x_true_opt=  X[index_true_opt,:]; y_true_opt= Y[index_true_opt,:]


    budget_init= 0
    print('initial budget:{}'.format(budget_init))
    if budget_init>0:
        X_init, Y_cost_init, Y_init, c= initial_evaluations(X, Y, Y_cost, cost_kernel, budget, budget_init, index0, noise_cost)

        print('X_init shape:{}, c:{}'.format(X_init.shape, c))
        Xt = X_init.copy()
        Yt = Y_init.copy()
        Yt_cost= Y_cost_init.copy()

    else:
        Xt= np.zeros([0, X.shape[1]]); Yt= np.zeros([0, Y.shape[1]]); Yt_cost= np.zeros([0, Y_cost.shape[1]])
        c=0

    if Yt.shape[0]>0:
        f_best= np.max(Yt)
        print('f_best:{}'.format(f_best))

    model= gp.models.GPR((Xt,Yt), kernel= kernel)
    model.likelihood.variance.assign(noise)

    if Xt.shape[0]>0:
        # warp_f = GPy.util.warping_functions.LogFunction()
        #
        # warped_cost = GPy.models.WarpedGP(Xt, Yt_cost, kernel_cost, warp_f)
        # warped_cost.likelihood.variance = noise_cost

        log_Yt_cost= np.log(Yt_cost);
        latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

    loss_list= []
    cost_list= []
    cum_cost_list= []
    '''Run BO'''

    C=0
    C+= c
    t=0

    while C<budget:
        print('t:',t)

        '''Evaluation'''
        if Xt.shape[0]==0:
            xt= X[index0,:].reshape(1,-1); yt= Y[index0,:].reshape(1,-1); yt_cost= Y_cost[index0,:].reshape(1,-1); \
                    xt_cost= X[index0,:].reshape(1,-1)
            f_best= yt

        else:
            u_X, var_X= model.predict_f(X)
            u_latent_cost, var_latent_cost= latent_cost_model.predict_f(X); sigma_latent_cost= np.sqrt(var_latent_cost)

            u_cost= np.exp(u_latent_cost);

            sigma_X= np.sqrt(var_X);

            Acq_EI= EI(sigma_X, u_X , f_best);

            Acq_EI_pu= EI_per_cost(sigma_X, u_X, f_best, u_cost)

            Acq_carbo= EI_cool(sigma_X, u_X , f_best, u_cost, C, budget, budget_init)

            # Acq_utlc= ucb_target_lcb_cost(u_X, sigma_X, u_latent_cost, sigma_latent_cost, 3.0, 1.0, f_best)

            # Acq= sum_of_std_weighted_ei_per_cost(u_X, sigma_X, u_cost, sigma_cost, f_best, )

            # Acq_pu_with_cost_exploration = EI_pu_with_cost_exploration(u_X, sigma_X, u_latent_cost, sigma_latent_cost,
            #                                    f_best, cost_kernel, noise_cost, Xt, Yt_cost, X, kapa_cost)

            # Acq= thompson_sample_cost(sigma_X, u_X, f_best, X, Xt, Yt_cost, cost_kernel, noise_cost, num_samp)

            Acq=  one_step_lookahead(sigma_X, u_X, u_cost, f_best, num_samp, X,  Xt, Yt, model, kernel, noise)



            Acq_dict= {'one_step_lookahead':Acq, 'carbo':Acq_carbo, 'EI_pu': Acq_EI_pu, 'EI': Acq_EI}

            index_max=  np.argmax(Acq,axis=0)

            xt= X[index_max,:]; yt= Y[index_max,:]; xt_cost= X[index_max,:]; yt_cost= Y_cost[index_max,:]

            if  yt<f_best:
                f_best= yt

        '''cost and target posterior plots'''
        if t > 0:

            if plot == True:
                if X.shape[1] == 1:
                    # posterior_acquisition_chosen_point_evaluated_points(X, Y, Xt, Yt, xt, yt, u_X, sigma_X, Acq, Acq_EI, Acq_EI_pu)

                    plot_posterior_and_true_target_and_acqusitions(X, Y, Y_cost, Xt, Yt, Yt_cost, xt,
                           u_X.numpy(), sigma_X, u_latent_cost.numpy(), sigma_latent_cost, Acq_dict,index_max, plot_cost)

                    input('press a key to continue')

                elif X.shape[1] == 2:
                    plot_posterior_and_true_target(X, u_X, Y, Xt, Yt, index_max)
                    input('press a key to continue')

            if plot_cost == True:

                # if X.shape[1] == 1:
                #     cost_posterior_chosen_point_evaluated_points(X, Y_cost, Xt, Yt_cost, xt_cost, yt_cost, u_cost,
                #                                                  sigma_cost)
                #     input('press a key to continue')

                if X.shape[1] == 2:
                    plot_cost_posterior_and_true_cost(X, u_cost, Y_cost, Xt, Yt_cost, index_max)
                    input('press a key to continue')


        Xt = np.append(Xt, xt, axis=0)
        Yt = np.append(Yt, yt, axis=0)
        Yt_cost= np.append(Yt_cost, yt_cost, axis=0)


        model = gp.models.GPR((Xt, Yt), kernel=kernel)
        model.likelihood.variance.assign(noise)

        # warp_f = GPy.util.warping_functions.LogFunction()
        # warped_cost= GPy.models.WarpedGP(Xt, Yt_cost, kernel_cost, warp_f)
        # warped_cost.likelihood.variance = noise_cost

        log_Yt_cost= np.log(Yt_cost);
        latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), cost_kernel)
        latent_cost_model.likelihood.variance.assign(10**(-4))

        loss, index_pred_opt, x_pred_opt, y_pred_opt= loss_at_current_step(X, Y, model, x_true_opt, y_true_opt)
        loss_list.append(loss)
        cost_list.append(yt_cost)
        '''increment the counter'''

        C+= yt_cost
        print('total cost:', C)
        cum_cost_list.append(C.copy())

        t+= 1


    if plot==True and X.shape[1]==2:

        plot_loss_vs_time(loss_list, cum_cost_list)
        # plot_evaluated_points(Xt, Yt, X, Y)
        # plot_posterior(X,Y, model)


    return loss_list, Xt, Yt, model, cost_list, cum_cost_list


def test_one_step_lookahead_bo(dimension, ls, var, ls_cost, var_cost,
                                     rang, disc, kernel_type, cost_kernel_type, budget, index0, kapa_cost, num_samp):
    # dimension= 1
    # ls= 1.0; var= 1.0; ls_cost= 0.5; var_cost= 1.0

    with h5py.File('../datasets/{}d/gp_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.\
                           format(dimension, kernel_type, ls, var, rang, disc), 'r') as hf:
        X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

    with h5py.File('../datasets/cost_{}d/gp_cost_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.\
                           format(dimension, cost_kernel_type, ls_cost, var_cost, rang, disc), 'r') as hf:
        X_cost= np.array(hf.get('X')); Y_cost= np.array(hf.get('y_sample'))
    #
    kernel= gp.kernels.RBF(lengthscales= [ls], variance= var)
    cost_kernel= gp.kernels.RBF(lengthscales= [ls_cost], variance= var_cost)
    noise_cost= 10**(-4)


    loss_list, Xt, Yt, model, cost_list, cum_cost_list= one_step_lookahead_bo(X, Y, Y_cost, kernel, cost_kernel, kapa_cost,
                           budget, index0, num_samp, noise=10**(-4), noise_cost= 10**(-4), plot=True, plot_cost=True)

    return loss_list, Xt, Yt, model, cost_list, cum_cost_list


dimension= 1; kernel_type= 'rbf'; cost_kernel_type= 'rbf'; ls= 0.5; ls_cost= 1.0; var= 1.0; var_cost= 1.0;
rang= 10;  disc= 100; budget= 5; num_samp= 10; index0= np.random.choice(np.arange(disc)); kapa_cost= 1.0

loss_list, Xt, Yt, model, cost_list, cum_cost_list= \
    test_one_step_lookahead_bo(dimension, ls, var, ls_cost, var_cost,
             rang, disc, kernel_type, cost_kernel_type, budget, index0, kapa_cost, num_samp)