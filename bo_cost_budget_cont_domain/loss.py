
import numpy as np
from scipy.optimize import minimize
# from scipy.optimize import shgo
# from scipy.optimize import dual_annealing
# from scipy.optimize import differential_evolution


def loss_at_current_step_cont_domain(model, x_true_opt, y_true_opt, domain, xt, yt, objective_func, X, u_X, D, random_restarts= 10,
                                     keras_model =None, num_layer= None, num_dense= None, num_epoch= None):

    '''Get posterior mean and variance'''
    # model= gp.models.GPR((Xt,Yt),kernel= kern_temp)

    def func(x):
        x= x.reshape(1,-1)
        u_x, var_x= model.predict_f(x);
        u_x=u_x.numpy()
        u_x= u_x.flatten()
        return u_x

    if type(u_X)== np.ndarray:
        index_opt_grid= int(np.argmin(u_X, axis=0));
        x_grid_opt= X[index_opt_grid, :].reshape(1,-1)
        x_opt_grid_val= u_X[index_opt_grid, :].reshape(1,-1)

        # x_pred_opt= shgo(func, bounds= domain)['x']
        # x_pred_opt= dual_annealing(func, bounds= domain)['x']
        # x_pred_opt= differential_evolution(func, bounds= domain)['x']

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    x_pred_list= np.zeros([random_restarts, D])
    x_pred_value_list= np.zeros([random_restarts, 1])

    for i in range(random_restarts):

        x0 = np.random.uniform(lower, upper, (1, D))
        result=  minimize(func, x0, bounds=domain, method='L-BFGS-B')
        x_pred = result['x']; x_pred_value= result['fun']
        x_pred= x_pred.reshape(1,-1); x_pred_value= x_pred_value.reshape(1,-1)
        x_pred_value_list[i, :]= x_pred_value[0, :]
        x_pred_list[i,:]= x_pred[0,:]


    index_opt_sci= np.argmin(x_pred_value_list, axis=0)
    x_pred_opt_value_sci= x_pred_value_list[index_opt_sci, :].reshape(1,-1)
    x_pred_opt_sci= x_pred_list[index_opt_sci, :].reshape(1, -1)

    if D==1 or D==2:
        '''compare with grid optimum'''
        if type(u_X)== np.ndarray:
            if x_pred_opt_value_sci< x_opt_grid_val:
                x_pred_opt = x_pred_opt_sci

            else:
                x_pred_opt = x_grid_opt
        else:
            x_pred_opt = x_pred_opt_sci

    else:
        x_pred_opt=  x_pred_opt_sci

    '''Find estimated optimal point from posterior'''

    if objective_func!= 'cifar' and objective_func!= 'fashion' and objective_func!= 'fashion2':
        y_pred_opt = objective_func(x_pred_opt)


        # print('u_opt_grid:{}, u_opt_sci:{}'.format(x_opt_grid_val, x_pred_opt_value_sci))

        '''Determine the squared loss for the prediction'''
        print('evaluated point at this round:{}'.format(xt))
        print('optimum predicted design point:{}, objective value at predicted optimum:{}'.format(x_pred_opt, y_pred_opt))
        print('true optimum design point:{}, objective value at optimum:{}'.format(x_true_opt, y_true_opt))
        loss = np.abs(float(y_pred_opt - y_true_opt))
        print('loss of predicted point:', loss)

    elif objective_func== 'cifar':
        print('u_opt_sci:{}'.format(x_pred_opt_value_sci))
        print('evaluated point at this round:{}'.format(xt))
        print('test error of keras model for the chosen point:{}'.format(yt))
        z = num_layer + num_dense

        filter_sizes = x_pred_opt[0, 0:num_layer]; dense_sizes = x_pred_opt[0, num_layer:z]; alpha = x_pred_opt[0, z];
        l2_regul = x_pred_opt[0, z+1]; dropout = x_pred_opt[0, z+2]

        y_pred_opt, y_pred_opt_cost, x_pred_opt= keras_model.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul, dropout,
                                                                         num_epoch= num_epoch)
        print('optimum predicted design point:{}, objective value at predicted optimum:{}'.format(x_pred_opt, y_pred_opt))
        loss = float(1+y_pred_opt)
        print('loss of predicted point:', loss)

    elif objective_func == 'fashion':

        print('u_opt_sci:{}'.format(x_pred_opt_value_sci))
        print('evaluated point at this round:{}'.format(xt))
        print('test error of keras model for the chosen point:{}'.format(yt))

        layer_sizes = x_pred_opt[0, 0:num_layer]; alpha = x_pred_opt[0, num_layer]; l2_regul = x_pred_opt[0, num_layer+1];
        # num_epoch= x_pred_opt[0, num_layer+ 2];

        y_pred_opt, y_pred_opt_cost,  x_pred_opt = keras_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul,  num_epoch= num_epoch)

        print('optimum predicted design point:{}, objective value at predicted optimum:{}'.format(x_pred_opt, y_pred_opt))

        loss = float(1+y_pred_opt)
        print('loss of predicted point:', loss)

    elif objective_func == 'fashion2':

        print('u_opt_sci:{}'.format(x_pred_opt_value_sci))
        print('evaluated point at this round:{}'.format(xt))
        print('test error of keras model for the chosen point:{}'.format(yt))

        layer_size = x_pred_opt[0,0]; num_layers= x_pred_opt[0,1]; alpha = x_pred_opt[0, 2];

        y_pred_opt, y_pred_opt_cost,  x_pred_opt = keras_model.evaluate_error_and_cost(layer_size, num_layers,  alpha,
                                                                                       num_epoch= num_epoch)

        print('optimum predicted design point:{}, objective value at predicted optimum:{}'.format(x_pred_opt, y_pred_opt))

        loss = float(1+y_pred_opt)
        print('loss of predicted point:', loss)

    return loss, x_pred_opt, y_pred_opt