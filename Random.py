import numpy as np
import gpflow as gp
from scipy.stats import norm
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
import sys
# sys.path.append('./examples')
# from branin import branin
from mpl_toolkits.mplot3d import Axes3D
import h5py
from plots import *




# rang= 3; disc= 20

# x1= np.linspace(0,rang,rang+1)-rang/2; x2= np.linspace(0,rang,rang+1)-rang/2
# x1= np.linspace(0,rang,disc)-rang/2; x2= np.linspace(0,rang,disc)-rang/2
# X1,X2= np.meshgrid(x1,x2); X1,X2= X1.flatten(), X2.flatten(); X1, X2= X1.reshape(-1,1), X2.reshape(-1,1)
# X= np.append(X1,X2, axis=1)
# Y= branin(X)

'''Load data from dataset'''
with h5py.File('./datasets/gp_sample.h5', 'r') as hf:
    X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

X1, X2= X[:,0].reshape(-1,1), X[:,1].reshape(-1,1)
# plt.scatter(X1,X2, c= y)
# plt.colorbar()
# fig = plt.figure()
# ax =  Axes3D(fig)
# ax.scatter3D(X[:,0], X[:,1], Y)

'''Estimate hyperparameters for this dataset by optimizing log-marginal through gradient ascent'''
kern_temp= gp.kernels.RBF()
model_temp= gp.models.GPR((X,Y), kernel= kern_temp)
gp.set_trainable(model_temp.likelihood, False)
model_temp.likelihood.variance.assign(10**(-4))
opt = gp.optimizers.Scipy()
opt_logs = opt.minimize(model_temp.training_loss, model_temp.trainable_variables, options=dict(maxiter=1000))
print_summary(model_temp)

'''specify budget and input-output dimensions. First evaluation point can be sampled randomly since the mean function is constant'''
budget= 50; d=2; m=1;t= 0
index0= np.random.choice(X.shape[0]); Xt= np.zeros([0,d]); Yt= np.zeros([0,m])


# '''Create gp model and use estimated hyperparameters in the model'''
# kern= gp.kernels.RBF(lengthscales=[1.0], variance=1.0)
# model= gp.models.GPR((np.zeros([0,d]), np.zeros([0,m])),kernel= kern_temp)
# model.likelihood.variance.assign(10**(-4))

def loss_at_this_step(model, x_true_opt, y_true_opt):
    '''Get posterior mean and variance'''
    # model= gp.models.GPR((Xt,Yt),kernel= kern_temp)
    u_X, sigma_X = model.predict_f(X)

    '''Find estimated optimal point from posterior'''
    index_pred_opt = np.argmin(u_X)
    x_pred_opt = X[index_pred_opt, :];
    y_pred_opt = Y[index_pred_opt, :]

    '''Determine the squared loss for the prediction'''
    print('optimum predicted design point:{}, objective value at predicted optimum:{}'.format(x_pred_opt, y_pred_opt))
    print('true optimum design point:{}, objective value at optimum:{}'.format(x_true_opt, y_true_opt))
    loss = np.square(float(y_pred_opt - y_true_opt))
    print('loss of predicted point:', loss)

    return loss, index_pred_opt, x_pred_opt, y_pred_opt

'''Find true optimal point'''
index_true_opt= np.argmin(Y,axis=0)
x_true_opt=  X[index_true_opt,:]; y_true_opt= Y[index_true_opt,:]

loss_list= []
index_pred_opt_list= []

'''Run BO'''
while t<budget:
    print('t:',t)

    '''Evaluation'''
    if t==0:
        xt= X[index0,:].reshape(1,-1); yt= Y[index0,:].reshape(1,-1)
        f_best= yt

    else:
        u_X, sigma_X= model.predict_f(X)
        # Acq= EI(sigma_X, u_X , f_best); index_max= np.argmax(Acq,axis=0)
        index_chosen= np.random.choice(np.arange(X.shape[0]))
        xt= X[index_chosen,:].reshape(1,-1); yt= Y[index_chosen,:].reshape(1,-1)

        if  yt<f_best:
            f_best= yt

    Xt = np.append(Xt, xt, axis=0)
    Yt = np.append(Yt, yt, axis=0)

    '''Update model'''
    model= gp.models.GPR((Xt,Yt),kernel= kern_temp); model.likelihood.variance.assign(10**(-4))
    loss, index_pred_opt, x_pred_opt, y_pred_opt= loss_at_this_step(model, x_true_opt, y_true_opt)
    loss_list.append(loss)

    '''increment time'''
    t+=1

plot_loss_vs_time(loss_list)
plot_evaluated_points(Xt, Yt, X, Y)
