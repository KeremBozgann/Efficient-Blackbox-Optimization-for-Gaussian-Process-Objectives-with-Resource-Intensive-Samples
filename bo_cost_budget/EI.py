import gpflow as gp
from scipy.stats import norm
# sys.path.append('./examples')
# from branin import branin
from mpl_toolkits.mplot3d import Axes3D
from plots import *
import GPy
import h5py

def EI(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))


    return EI_x

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

def EI_bo(X, Y, Y_cost, kernel, budget, index0, cost_kernel, noise=10**(-4), noise_cost= 10**(-4), plot=False, plot_cost= False):

    '''Find true optimal point'''
    index_true_opt= np.argmin(Y,axis=0)
    x_true_opt=  X[index_true_opt,:]; y_true_opt= Y[index_true_opt,:]


    Xt = np.zeros([0, X.shape[1]])
    Yt = np.zeros([0, Y.shape[1]])
    Yt_cost= np.zeros([0,Y_cost.shape[1]])

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

    while C<budget:
        print('t:',t)

        '''Evaluation'''
        if t==0:
            xt= X[index0,:].reshape(1,-1); yt= Y[index0,:].reshape(1,-1); yt_cost= Y_cost[index0,:].reshape(1,-1); \
                    xt_cost= X[index0,:].reshape(1,-1)
            f_best= yt

        else:
            u_X, var_X= model.predict_f(X)
            u_latent_cost, var_latent_cost= latent_cost_model.predict_f(X); sigma_latent_cost= np.sqrt(var_latent_cost)
            u_cost= np.exp(u_latent_cost)
            sigma_X= np.sqrt(var_X);


            Acq= EI(sigma_X, u_X , f_best); index_max= np.argmax(Acq,axis=0)
            xt= X[index_max,:]; yt= Y[index_max,:]; xt_cost= X[index_max,:]; yt_cost= Y_cost[index_max,:]

            Acq_dict= {'EI': Acq}

            if  yt<f_best:
                f_best= yt

        '''cost and target posterior plots'''
        if t > 0:

            if plot == True:
                if X.shape[1] == 1:
                    plot_posterior_and_true_target_and_acqusitions(X, Y, Y_cost, Xt, Yt, Yt_cost, xt,
                                                                   u_X.numpy(), sigma_X, u_latent_cost, sigma_latent_cost,
                                                                   Acq_dict, index_max, plot_cost)
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

        log_Yt_cost= np.log(Yt_cost)
        latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

        loss, index_pred_opt, x_pred_opt, y_pred_opt= loss_at_current_step(X, Y, model, x_true_opt, y_true_opt)
        loss_list.append(loss)
        cost_list.append(yt_cost)
        '''increment the counter'''

        C+= yt_cost
        print('total cost:', C)
        cum_cost_list.append(C.copy())

        t+= 1


    if plot==True and X.shape[1]==2:

        plot_loss_vs_time(loss_list)
        plot_evaluated_points(Xt, Yt, X, Y)
        plot_posterior(X,Y, model)


    return loss_list, Xt, Yt, model, cost_list, cum_cost_list
#


def test_ei(dimension, kern_type, kern_type_cost, ls, ls_cost, var, var_cost, rang, disc, budget, index0):

    with h5py.File('../datasets/{}d/gp_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(dimension, kern_type, ls, var, rang, disc), 'r') as hf:
        X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))



    with h5py.File('../datasets/cost_{}d/gp_cost_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'. \
                           format(dimension, kern_type_cost, ls_cost, var_cost, rang, disc), 'r') as hf:
        X_cost= np.array(hf.get('X')); Y_cost= np.array(hf.get('y_sample'))
    #

    kernel= gp.kernels.RBF(lengthscales= [ls], variance= var)
    cost_kernel= gp.kernels.RBF(lengthscales= [ls_cost], variance= var_cost)


    loss_list, Xt, Yt, model, coss_list, cum_cost_list= \
        EI_bo(X, Y, Y_cost, kernel,  budget, index0, cost_kernel, noise= 10**(-4), noise_cost= 10**(-4), plot= True, plot_cost= True)

    return loss_list, Xt, Yt, model, coss_list, cum_cost_list

#
# dimension= 1; kern_type= 'rbf'; kern_type_cost= 'rbf'; ls= 0.5; ls_cost= 1.0; var= 1.0; var_cost= 1.0;
# rang= 10;  disc= 100; budget= 5; index0= np.random.choice(np.arange(disc))
#
# loss_list, Xt, Yt, model, coss_list, cum_cost_list=  \
#     test_ei(dimension, kern_type, kern_type_cost, ls, ls_cost, var, var_cost, rang, disc, budget, index0)