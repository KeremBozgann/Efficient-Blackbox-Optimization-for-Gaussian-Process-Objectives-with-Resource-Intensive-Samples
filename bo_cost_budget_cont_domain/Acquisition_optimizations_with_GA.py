
import numpy as np

from gp_gradients import dEI_dx

from Acquisitions import EI

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize
from scipy.optimize import Bounds

import time


def grad_ascent(x, grad_x,  domain, D , t, alpha= 4):

    flag =0

    x_new= x+ grad_x*(alpha/(alpha+t))


    for i in range(D):
        if domain[i][0]> x_new[0, i] or domain[i][1]< x_new[0, i]:
            flag= 1
            break

    if flag==1:
        x_new= x.copy()

    return x_new, flag


def EI_optimize(model, noise, x, kernel, Xt, Yt, f_best, num_restarts, domain, D, num_iter=100):

    # u_x, var_x = model.predict_f(x); u_x = u_x.numpy(); var_x = var_x.numpy()
    # sigma_x= np.sqrt(var_x)

    def EI_negative_gradient(x):

        x = x.reshape(1, -1)

        u_x, var_x = model.predict_f(x); u_x = u_x.numpy(); var_x = var_x.numpy();
        sigma_x = np.sqrt(var_x)


        def _EI_negative_gradient(u_x, sigma_x, kernel, Xt, Yt, x ,noise):



            grad_x= dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise)

            grad_x= grad_x.flatten()

            return -grad_x

        return _EI_negative_gradient(u_x, sigma_x, kernel, Xt, Yt, x ,noise)

    def EI_negative(x):

        x = x.reshape(1, -1)
        u_x, var_x = model.predict_f(x); u_x = u_x.numpy(); var_x = var_x.numpy();
        sigma_x = np.sqrt(var_x)

        def _EI_negative(x, sigma_x, u_x, f_best):


            EI_x= EI(sigma_x, u_x, f_best)

            EI_x= EI_x.flatten()

            return -EI_x

        return _EI_negative(x, sigma_x, u_x, f_best)


    lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]

    '''optimization with grad_ascent'''
    # x_list= np.empty([num_restarts, x.shape[1]])
    x_list= np.empty([num_restarts, x.shape[1]])
    value_list= np.empty([num_restarts, 1])
    grad_list= np.empty([num_restarts, x.shape[1]])

    x_all_list= []
    x0_list= np.zeros([num_restarts, D])

    t1_ga= time.process_time_ns()

    for i in range(num_restarts):

        x= np.random.uniform(lower, upper, (1,D))
        x0_list[i, :] = x[0, :]
        xi_list= np.zeros([0, D])

        for k in range(num_iter):


            grad_x= -EI_negative_gradient(x)
            x, flag= grad_ascent(x, grad_x, domain ,D, k)
            # print('x, grad_x', x, grad_x)
            xi_list= np.append(xi_list, x, axis=0)

            if flag==1:
                break

        x_all_list.append(xi_list)

        value_x= -EI_negative(x); value_x= value_x.reshape(1,-1)
        grad_x = -EI_negative_gradient(x); grad_x= grad_x.reshape(1,-1)

        x_list[i, :]= x[0,:]
        value_list[i,:]= value_x[0,:]
        grad_list[i,:] = grad_x[0,:]

    t2_ga= time.process_time_ns()
    t_ga= t2_ga- t1_ga

    index_max= int(np.argmax(value_list, axis= 0))
    x_opt= x_list[index_max, :].reshape(1,-1)
    value_opt= value_list[index_max, :].reshape(1,-1)
    grad_opt= grad_list[index_max, :].reshape(1,-1)

    '''optimization with scipy'''
    lower= []; upper= []
    for i in range(len(domain)):
        lower.append(domain[i][0])
        upper.append(domain[i][1])
    b= Bounds(lb= lower, ub= upper )

    x_opt_list_sci= np.zeros([num_restarts, D])
    x_value_list_sci= np.zeros([num_restarts, 1])
    x_grad_sci_list= np.zeros([num_restarts, D])

    t1_sci= time.process_time_ns()

    for i in range(num_restarts):

        x0 = x0_list[i, :].reshape(1,-1)
        result= minimize(EI_negative, x0, bounds=b, method= 'L-BFGS-B', jac= EI_negative_gradient, options= {'maxiter': num_iter})
        x_sci= result['x']; x_grad_sci= -result['jac']; x_value_sci= -result['fun']
        x_sci = x_sci.reshape(1,-1); x_grad_sci= x_grad_sci.reshape(1,-1); x_value_sci= x_value_sci.reshape(1,-1)

        x_opt_list_sci[i, :]= x_sci[0,:]
        x_value_list_sci[i,:]= x_value_sci[0,:]
        x_grad_sci_list[i,:] = x_grad_sci[0,:]

    index_opt_sci= np.argmax(x_value_list_sci, axis=0)
    x_opt_sci= x_opt_list_sci[index_opt_sci, :].reshape(1,-1)
    x_opt_value_sci= x_value_list_sci[index_max, :].reshape(1,-1)
    x_opt_grad_sci= x_grad_sci_list[index_opt_sci, :].reshape(1,-1)

    t2_sci= time.process_time_ns()
    t_sci= t2_sci- t1_sci

    print('x_value_sci:{}, x_value_GA:{}'.format(x_opt_value_sci, value_opt))
    print('sci_time:{}(s), ga_time:{}(s)'.format(t_sci*10**(-9), t_ga*10**(-9)))
    print('x_sci:{}, x_ga:{}'.format(x_opt_sci, x_opt))

    return x_opt, value_opt, grad_opt, x_all_list, x_list, value_list, grad_list, x0_list



def comparison_of_optimum_with_grid(f_best, model, domain, kernel, Xt, Yt, D, noise= 10**(-4)):

    num_restarts= 3

    '''compare with grid'''
    x_grid1 = np.linspace(domain[0][0], domain[0][1], 101);

    X_grid = x_grid1.reshape(-1,1)

    u_X, var_X= model.predict_f(X_grid);
    u_X = u_X.numpy(); var_X = var_X.numpy(); sigma_X= np.sqrt(var_X)

    EI_X= EI(sigma_X, u_X, f_best)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]
    x0= np.random.uniform(lower, upper, (1,D))

    x_pred_opt, x_pred_opt_val, grad_pred_opt, x_all_list, x_list, value_list, grad_list, x0_list= \
        EI_optimize(model, noise, x0, kernel, Xt, Yt, f_best, num_restarts, domain, D, num_iter=100)

    if D==1:
        plt.figure()
        plt.plot(X_grid[:,0], EI_X[:,0])
        plt.scatter(x_pred_opt[0,:], x_pred_opt_val[0,:], color= 'blue')
        plt.scatter((x_all_list[0])[:,0], np.zeros([x_all_list[0].shape[0], ]), color= 'orange', alpha= 0.5)
        plt.scatter(x0_list[:,0], np.zeros([x0_list.shape[0], ]), color= 'red', alpha= 0.5)

        plt.show()

def comparison_of_optimum_with_grid_2d(f_best, model, domain, kernel, Xt, Yt, D, noise=10 ** (-4)):

    num_restarts = 3

    '''compare with grid'''
    x_grid1 = np.linspace(domain[0][0], domain[0][1], 21);
    x_grid2=  np.linspace(domain[1][0], domain[1][1], 21);
    X1, X2= np.meshgrid(x_grid1, x_grid2); X1 = (X1.flatten()).reshape(-1,1); X2= (X2.flatten()).reshape(-1,1)
    X_grid = np.append(X1, X2, axis=1)

    u_X, var_X = model.predict_f(X_grid);
    u_X = u_X.numpy();
    var_X = var_X.numpy();
    sigma_X = np.sqrt(var_X)

    EI_X = EI(sigma_X, u_X, f_best)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]
    x0 = np.random.uniform(lower, upper, (1, D))

    x_pred_opt, x_pred_opt_val, grad_pred_opt, x_all_list, x_list, value_list, grad_list, x0_list = \
        EI_optimize(model, noise, x0, kernel, Xt, Yt, f_best, num_restarts, domain, D, num_iter=100)


    if D == 2:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(X_grid[:,0], X_grid[:,1], EI_X[:,0], color= 'red', alpha= 0.5)
        ax.scatter3D(x_pred_opt[0,0], x_pred_opt[0,1], x_pred_opt_val[0,0], color= 'green')
        ax.scatter3D((x_all_list[0])[:, 0], (x_all_list[0])[:, 1], np.zeros([x_all_list[0].shape[0], ]), color='orange', alpha=0.5)
        plt.show()


    # if D==2:

