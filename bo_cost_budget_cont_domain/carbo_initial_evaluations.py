from scipy.spatial import distance
import gpflow as gp
import numpy as np
from plots import *
import sys
sys.path.append('../HPO')
from keras_model import Keras_model_cifar, Keras_model_fashion

def update_cost_model(X_init, Y_cost_init, cost_kernel, noise_cost):

    if X_init.shape[0]> 0:
        log_Y_cost_init= np.log(Y_cost_init)
        latent_cost_model= gp.models.GPR((X_init,log_Y_cost_init), cost_kernel)
        latent_cost_model.likelihood.variance.assign(noise_cost)

        return latent_cost_model

    else:
        return None

def get_distance_to_X_init(temp_X, X_init):
    if X_init.shape[0]==0:
        dist= np.ones(temp_X.shape[0])
    else:
        dist_matrix= distance.cdist(temp_X, X_init)
        dist= np.min(dist_matrix, axis=1)

    return dist

def get_chosen_index(X_cand, x_chosen):

    row_index= np.where((X_cand == x_chosen).all(axis=1))[0]

    return row_index

def evaluate_chosen(x_chosen, objective_function, cost_function, num_layer, num_dense, num_epoch):

    if objective_function == 'fashion':

        fashion_model= Keras_model_fashion()
        layer_sizes = x_chosen[0, 0:num_layer];
        print(layer_sizes)
        alpha = x_chosen[0, num_layer];
        l2_regul = x_chosen[0, num_layer + 1];
        # num_epoch = x_chosen[0, num_layer + 2]
        yt, yt_cost, x_chosen = fashion_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul,  num_epoch)

    elif objective_function == 'cifar':

        cifar_model= Keras_model_cifar()
        z = num_layer + num_dense;
        filter_sizes = x_chosen[0, 0:num_layer];
        dense_sizes = x_chosen[0, num_layer:z];
        alpha = x_chosen[0, z];
        l2_regul = x_chosen[0, z + 1];
        dropout = x_chosen[0, z + 2]

        yt, yt_cost, x_chosen  = cifar_model.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul, dropout,
                                                          num_epoch= num_epoch)

    else:
        yt = objective_function(x_chosen);
        yt_cost = cost_function(x_chosen)

    return yt, yt_cost, x_chosen


def initial_evaluations(X, cost_function, objective_function, cost_kernel, budget_init, noise_cost,
                        D, num_layer= None, num_dense= None, num_epoch= 1.7):

    # X_init= x0.copy(); Y_cost_init= cost_function(X_init); Y_init= objective_function(X_init)

    X_init= np.zeros([0, D]); Y_init= np.zeros([0, 1]); Y_cost_init= np.zeros([0, 1])

    # Y_cost= cost_function(X); Y= objective_function(X)

    X_cand= X.copy(); #Y_cost_cand= Y_cost.copy(); Y_cand= Y.copy()

    c=0

    t=0

    while c<budget_init :

        if t>0:
            X_init = np.append(X_init, x_chosen, axis= 0)
            Y_cost_init= np.append(Y_cost_init, y_cost_chosen, axis=0)
            Y_init = np.append(Y_init, y_chosen, axis=0)

            index_chosen= get_chosen_index(X_cand, x_chosen)
            X_cand= np.delete(X_cand, index_chosen, axis=0)
            # Y_cost_cand= np.delete(Y_cost_cand, index_chosen, axis=0)
            # Y_cand = np.delete(Y_cand, index_chosen, axis=0)

        if X_cand.shape[0]==1:
            x_chosen = X_cand.copy();
            y_chosen, y_cost_chosen, x_chosen = evaluate_chosen(x_chosen, objective_function, cost_function, num_layer, num_dense,  num_epoch)
            # y_cost_chosen = Y_cost_cand.copy();
            # y_chosen =Y_cand.copy()

            X_init= np.append(X_init, x_chosen, axis=0);
            Y_cost_init= np.append(Y_cost_init, y_cost_chosen, axis=0)
            Y_init= np.append(Y_init, y_chosen, axis=0)

            c+= y_cost_chosen

            break

        if t==0:
            u_latent_cost= cost_kernel.variance.numpy()*np.ones(X_cand.shape[0])
            u_cost= np.exp(u_latent_cost)

        else:
            latent_cost_model = update_cost_model(X_init, Y_cost_init, cost_kernel, noise_cost)
            u_latent_cost, var_latent_cost = latent_cost_model.predict_f(X_cand); u_cost= np.exp(u_latent_cost)

        temp_X= X_cand.copy(); #temp_Y_cost= Y_cost_cand.copy(); temp_Y= Y_cand.copy()

        while temp_X.shape[0]>1:

            '''exclude most cosly point'''
            index_max_cost= np.argmax(u_cost)
            temp_X= np.delete(temp_X, index_max_cost, axis=0);
            # temp_Y_cost= np.delete(temp_Y_cost, index_max_cost, axis=0)
            # temp_Y= np.delete(temp_Y, index_max_cost, axis=0)

            u_cost= np.delete(u_cost, index_max_cost, axis=0)

            if temp_X.shape[0]==1:
                break
            '''exclude closest point to X_init '''
            index_max_dist= np.argmax(get_distance_to_X_init(temp_X, X_init))
            temp_X = np.delete(temp_X, index_max_dist, axis=0);
            # temp_Y_cost= np.delete(temp_Y_cost, index_max_dist, axis=0)
            # temp_Y= np.delete(temp_Y, index_max_dist, axis=0)

            u_cost = np.delete(u_cost, index_max_dist, axis=0);

        x_chosen= temp_X.copy();
        y_chosen, y_cost_chosen, x_chosen = evaluate_chosen(x_chosen, objective_function, cost_function, num_layer, num_dense, num_epoch)

        c+= y_cost_chosen
        t+=1

    #if breaks after 1 evaluation
    if t==1:
        X_init = np.append(X_init, x_chosen, axis=0)
        Y_cost_init = np.append(Y_cost_init, y_cost_chosen, axis=0)
        Y_init = np.append(Y_init, y_chosen, axis=0)

    return X_init, Y_cost_init, Y_init, c