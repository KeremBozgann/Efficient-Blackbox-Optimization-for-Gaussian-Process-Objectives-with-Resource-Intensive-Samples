import gpflow as gp
import numpy as np
import tensorflow as tf
from plots import *

# def entropy_acquisition(X, Xt, Yt, kernel, model):
#
#     for i in range(X.shape[0]):
#
#         psedo_inp= np.append(Xt, X[i,:].reshape(1,-1), axis=0)
#         psedo_out= np.append(Yt, np.zeros(1,Yt.shape[1]),axis=0)
#         model_psed= gp.models.GPR((Xt, Yt),kernel=kernel)



def WEI(Xt, Yt, X,  kernel, num_samples, noise):

    model= gp.models.GPR((Xt, Yt),kernel=kernel)
    model.likelihood.variance.assign(noise)

    temp= model.predict_f_samples(X, num_samples); y_samp= tf.reshape(temp,(X.shape[0],num_samples))

    index_min_list= np.empty([X.shape[0],1])

    min_counter= np.zeros([X.shape[0], 1])

    for i in range(num_samples):

        yi= tf.reshape(y_samp[:,i], (-1,1))

        index_min= np.argmin(yi,axis=0)

        min_counter[index_min,0]+=1

        index_min_list[i,0]= index_min
        # x_min= X[index_min,:]; y_min= yi[index_min,:]

    MC_probabilities_of_being_minimum= min_counter/num_samples

    log_ent_reduct_list= np.empty([X.shape[0],1])

    for i in range(X.shape[0]):

        #x_psed is equal to x
        x_psed= X[i,:].reshape(1,-1); Xt_psed=  np.append(Xt,x_psed, axis=0)
        temp1= tf.matmul(tf.matmul(kernel.K(X,Xt),tf.linalg.inv(tf.add(kernel.K(Xt,Xt), tf.cast(noise*tf.eye(Xt.shape[0]),tf.float64)))), tf.transpose(kernel.K(X,Xt)))
        temp2= tf.matmul(tf.matmul(kernel.K(X,Xt_psed),tf.linalg.inv(tf.add(kernel.K(Xt_psed,Xt_psed), tf.cast(noise*tf.eye(Xt_psed.shape[0]),tf.float64)))), tf.transpose(kernel.K(X,Xt_psed)))

        # ent_prior= 1/2*(2*np.pi*np.e)**(X.shape[0])*np.linalg.det(kernel.K(X,X)- temp1); ent_psed= 1/2*(2*np.pi*np.e)**(X.shape[0])*np.linalg.det(kernel.K(X,X)- temp2)
        log_ent_prior= np.log(1/2)+ X.shape[0]*np.log(2*np.pi*np.e)+ np.log(np.linalg.det(kernel.K(X,X)- temp1))
        log_ent_psed= np.log(1/2)+ X.shape[0]*np.log(2*np.pi*np.e)+ np.log(np.linalg.det(kernel.K(X,X)- temp2))

        log_entropy_reduct_x= log_ent_prior- log_ent_psed

        log_ent_reduct_list[i,0]= (log_entropy_reduct_x)

    weighted_log_ent_imp= MC_probabilities_of_being_minimum* log_ent_reduct_list

    return weighted_log_ent_imp

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


import h5py

with h5py.File('./datasets/gp_sample.h5', 'r') as hf:
    X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

kernel= gp.kernels.RBF(); budget= 10; index0= np.random.choice(np.arange(X.shape[0]))
budget= 10; index0=  np.random.choice(np.arange(X.shape[0])); noise= 10**(-4); num_mc_samples= 100; plot=False

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
        u_X, sigma_X= model.predict_f(X)
        Acq= WEI(Xt, Yt, X, kernel, num_mc_samples, noise); index_max= np.argmax(Acq,axis=0)
        xt= X[index_max,:]; yt= Y[index_max,:]

        if  yt<f_best:
            f_best= yt

    Xt = np.append(Xt, xt, axis=0)
    Yt = np.append(Yt, yt, axis=0)

    model = gp.models.GPR((Xt, Yt), kernel=kernel)
    model.likelihood.variance.assign(10 ** (-4))

    loss, index_pred_opt, x_pred_opt, y_pred_opt= loss_at_current_step(X, Y, model, x_true_opt, y_true_opt)
    loss_list.append(loss)

    '''increment the counter'''
    t+=1

if plot==True:

    plot_loss_vs_time(loss_list)
    plot_evaluated_points(Xt, Yt, X, Y)
    plot_model_fit_to_data(X,Y, model)




