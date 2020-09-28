import gpflow as gp
from scipy.stats import norm
# sys.path.append('./examples')
# from branin import branin
from mpl_toolkits.mplot3d import Axes3D
from plots import *

def PI(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    PI_x = norm.cdf(gama_x)

    return PI_x

def loss_at_current_step(X, Y, model, x_true_opt, y_true_opt):
    '''Get posterior mean and variance'''
    # model= gp.models.GPR((Xt,Yt),kernel= kern_temp)
    u_X, var_X = model.predict_f(X)
    sigma_X = np.sqrt(var_X)
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

def PI_bo(X, Y, kernel, budget, index0, noise= 10**(-4), plot=False):

    '''Find true optimal point'''
    index_true_opt= np.argmin(Y,axis=0)
    x_true_opt=  X[index_true_opt,:]; y_true_opt= Y[index_true_opt,:]


    Xt = np.zeros([0, X.shape[1]]);
    Yt = np.zeros([0, Y.shape[1]])
    model= gp.models.GPR((Xt,Yt), kernel= kernel)
    model.likelihood.variance.assign(noise)

    loss_list= []

    '''Run BO'''

    t = 0

    while t<budget:
        print('t:',t)

        '''Evaluation'''
        if t==0:
            xt= X[index0,:].reshape(1,-1); yt= Y[index0,:].reshape(1,-1)
            f_best= yt

        else:

            u_X, var_X= model.predict_f(X)
            sigma_X = np.sqrt(var_X)

            Acq= PI(sigma_X, u_X , f_best); index_max= np.argmax(Acq,axis=0)
            xt= X[index_max,:]; yt= Y[index_max,:]

            if  yt<f_best:
                f_best= yt

        if X.shape[1] == 1 and t > 0 and plot == True:
            scatter_chosen_point_and_previously_evaluated_points_and_plot_posterior_plot_true_function(X, Y, Xt, Yt, xt, yt, u_X, sigma_X)
            input('press a key to continue')

        Xt = np.append(Xt, xt, axis=0)
        Yt = np.append(Yt, yt, axis=0)

        model = gp.models.GPR((Xt, Yt), kernel=kernel)
        model.likelihood.variance.assign(10 ** (-4))

        loss, index_pred_opt, x_pred_opt, y_pred_opt= loss_at_current_step(X, Y, model, x_true_opt, y_true_opt)
        loss_list.append(loss)

        '''increment the counter'''
        t+=1

    if plot==True and X.shape[1]==2:

        plot_loss_vs_time(loss_list)
        plot_evaluated_points(Xt, Yt, X, Y)
        plot_model_fit_to_data(X,Y, model)


    return loss_list, Xt, Yt, model

# import h5py
#
# with h5py.File('./datasets/1d/gp_sample_rbf_l_1.0_v_1.0.h5', 'r') as hf:
#     X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))
#
# kernel= gp.kernels.RBF(); budget= 10; index0= np.random.choice(np.arange(X.shape[0]))
# loss_list, Xt, Yt, model= PI_bo(X, Y, kernel, budget, index0, noise= 10**(-4))