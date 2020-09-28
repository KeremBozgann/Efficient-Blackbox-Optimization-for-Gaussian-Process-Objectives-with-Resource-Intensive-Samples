import numpy as np
import gpflow as gp
import h5py
from plots import *
import os
import tensorflow as tf

'''kern type and hyper parameters'''
# kern_type= 'matern52'
# ls= 5.0; var= 1.0

'''input space discretization parameters'''
# rang= 5; disc= 20

def gp_sample_2d(kern_type, ls, var, rang, disc, noise, save= False, plot=True ):

    x1= np.linspace(0,rang,disc)-rang/2; x2= np.linspace(0,rang,disc)-rang/2
    X1,X2= np.meshgrid(x1,x2); X1,X2= X1.flatten(), X2.flatten(); X1, X2= X1.reshape(-1,1), X2.reshape(-1,1)
    X= np.append(X1,X2, axis=1)

    if kern_type=='matern52':
        kern= gp.kernels.Matern52(lengthscales=[ls], variance= var)

    elif kern_type=='rbf':
        kern = gp.kernels.RBF(lengthscales=[ls], variance=var)


    model= gp.models.GPR((np.zeros([0,2]), np.zeros([0,1])),kernel=kern)
    model.likelihood.variance.assign(noise)

    y_sample = model.predict_f_samples(X); y_sample= y_sample.numpy()

    if save==True:
        file_name= 'gp_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(kern_type, ls, var)

        if not file_name in os.listdir('./datasets/2d'):

            with h5py.File('./datasets/2d/'+ file_name, 'w') as hf:
                hf.create_dataset('X',data= X); hf.create_dataset('y_sample', data= y_sample.numpy())

            with h5py.File('./datasets/2d/'+ file_name, 'r') as hf:
                X= np.array(hf.get('X')); y= np.array(hf.get('y_sample'))

        else:

            print('sample from gp with same type and hyperparameters already exists')

    if plot==True:
        plot_3D(X[:, 0], X[:, 1], y_sample[:, 0])

    return X, y_sample

def gp_sample_1d(kern_type, ls, var, rang, disc, noise, save= False, plot=True):

    x1 = np.linspace(0, rang, disc) - rang / 2;
    X= x1.reshape(-1,1)

    if kern_type == 'matern52':
        kern = gp.kernels.Matern52(lengthscales=[ls], variance=var)

    elif kern_type == 'rbf':
        kern = gp.kernels.RBF(lengthscales=[ls], variance=var)

    model = gp.models.GPR((np.zeros([0, 1]), np.zeros([0, 1])), kernel=kern)
    model.likelihood.variance.assign(noise)

    y_sample = model.predict_f_samples(X); y_sample= y_sample.numpy()

    if save==True:

        file_name = 'gp_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(kern_type, ls, var, rang,disc)

        if not file_name in os.listdir('./datasets/1d/'):

            with h5py.File('./datasets/1d/' + file_name, 'w') as hf:
                hf.create_dataset('X', data=X);
                hf.create_dataset('y_sample', data=y_sample.numpy())

            with h5py.File('./datasets/1d/' + file_name, 'r') as hf:
                X = np.array(hf.get('X'));
                y = np.array(hf.get('y_sample'))

        else:

            print('sample from gp with same type and hyperparameters already exists')

    if plot==True:
        plot_2D(X,y_sample)

    return X, y_sample

def gp_cost_sample_1d(kern_type, ls, var, rang, disc, noise, save= False, plot=True):

    x1 = np.linspace(0, rang, disc) - rang / 2;
    X= x1.reshape(-1,1)

    if kern_type == 'matern52':
        kern = gp.kernels.Matern52(lengthscales=[ls], variance=var)

    elif kern_type == 'rbf':
        kern = gp.kernels.RBF(lengthscales=[ls], variance=var)

    model = gp.models.GPR((np.zeros([0, 1]), np.zeros([0, 1])), kernel=kern)
    model.likelihood.variance.assign(noise)

    y_latent = model.predict_f_samples(X)

    y_sample= tf.exp(y_latent); y_sample= y_sample.numpy()

    if save==True:
        dimension= X.shape[1]
        file_name = 'gp_cost_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(kern_type, ls, var,rang,disc)


        if not file_name in os.listdir('./datasets/cost_1d/'):

            with h5py.File('./datasets/cost_1d/' + file_name, 'w') as hf:
                hf.create_dataset('X', data=X);
                hf.create_dataset('y_sample', data=y_sample)

            with h5py.File('./datasets/cost_1d/' + file_name, 'r') as hf:
                X = np.array(hf.get('X'));
                y = np.array(hf.get('y_sample'))

        else:

            print('sample from gp with same type and hyperparameters already exists')

    if plot==True:
        '''plot warped'''
        plot_2D(X,y_sample)

        '''plot latent'''
        plot_2D(X, y_latent)

    return X, y_sample

def gp_cost_sample_2d(kern_type, ls, var, rang, disc, noise, save= False, plot= True):

    x1= np.linspace(0,rang,disc)-rang/2; x2= np.linspace(0,rang,disc)-rang/2
    X1,X2= np.meshgrid(x1,x2); X1,X2= X1.flatten(), X2.flatten(); X1, X2= X1.reshape(-1,1), X2.reshape(-1,1)
    X= np.append(X1,X2, axis=1)

    if kern_type=='matern52':
        kern= gp.kernels.Matern52(lengthscales=[ls], variance= var)

    elif kern_type=='rbf':
        kern = gp.kernels.RBF(lengthscales=[ls], variance=var)


    model= gp.models.GPR((np.zeros([0,2]), np.zeros([0,1])),kernel=kern)
    model.likelihood.variance.assign(noise)

    y_latent = model.predict_f_samples(X)
    y_sample= tf.exp(y_latent); y_sample= y_sample.numpy()

    if save== True:

        file_name= 'gp_cost_sample_{}_l_{}_v_{}_rang_{}_disc{}.h5'.format(kern_type, ls, var)

        if not file_name in os.listdir('./datasets/cost_2d/'):

            with h5py.File('./datasets/cost_2d/'+ file_name, 'w') as hf:
                hf.create_dataset('X',data= X); hf.create_dataset('y_sample', data= y_sample.numpy())

            with h5py.File('./datasets/cost_2d/'+ file_name, 'r') as hf:
                X= np.array(hf.get('X')); y= np.array(hf.get('y_sample'))

        else:

            print('sample from gp with same type and hyperparameters already exists')

    if plot==True:
        plot_3D(X[:, 0], X[:, 1], y_sample[:, 0])

    return X, y_sample
#
# kern_type= 'rbf'; ls= 1.0; var= 1.0; rang=10; disc= 100; noise= 10**(-4)
# # gp_sample_1d(kern_type, ls, var, rang, disc)
# X, y_sample= gp_cost_sample_1d(kern_type, ls, var, rang, disc, noise, save= False, plot= True)
# print('y_sample min:{}, y_sample_max:{}'.format(np.min(y_sample), np.max(y_sample)))