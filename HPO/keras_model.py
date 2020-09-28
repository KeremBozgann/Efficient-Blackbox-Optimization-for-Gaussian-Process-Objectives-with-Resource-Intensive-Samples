import tensorflow as tf
from tensorflow import keras
import time

import gpflow as gp
from tensorflow.keras.regularizers import l2

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import to_categorical
from tensorflow.keras.utils import to_categorical
from gpflow.utilities import print_summary
import tensorflow_probability as tfp
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import load_img
import numpy as np


def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

def load_cifar():
	# load dataset
	(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY


class Keras_model_fashion():

    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        self.train_images= train_images
        self.train_labels= train_labels
        self.test_images= test_images
        self.test_labels= test_labels

    def evaluate_error_and_cost(self, layer_sizes, alpha, l2_regul, num_epoch):
        layer_sizes = np.ndarray.astype(np.rint(np.power(2, layer_sizes)), dtype= int)
        alpha= np.power(10.0, alpha)
        l2_regul= np.power(10.0, l2_regul)
        num_epoch= np.int(np.rint(np.power(10, num_epoch)))

        xt_eval= np.empty([1, len(layer_sizes)+ 2])

        xt_eval[0, 0:len(layer_sizes)]= np.log2(layer_sizes[:])
        # xt_eval[0,len(filter_sizes):len(filter_sizes)+ len(dense_sizes)]= np.log2(dense_sizes[:])
        xt_eval[0,len(layer_sizes)]= np.log10(alpha)
        xt_eval[0, len(layer_sizes)+1]= np.log10(l2_regul)

        print('layer_sizes:{}\nalpha:{}\nl2_regul:{}\nnum_epoch:{}'.format(layer_sizes, alpha, l2_regul, num_epoch))
        '''define model'''
        t1= time.clock()
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(layer_sizes[0], activation='relu',  kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[1], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[2], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[3], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(10)
        ])
        opt = keras.optimizers.Adam(learning_rate= alpha)
        '''compile model'''
        model.compile(optimizer= opt,  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        '''fit'''
        model.fit(self.train_images, self.train_labels, epochs= num_epoch)

        '''test set accuracy'''
        test_loss, test_acc = model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        t2= time.clock()

        return np.atleast_2d(-test_acc), np.atleast_2d(t2- t1), xt_eval



class Keras_model_cifar():

    def __init__(self):

        train_images, train_labels, test_images, test_labels= load_cifar()
        train_images,  test_images = prep_pixels( train_images, test_images)

        self.train_images= train_images
        self.train_labels= train_labels
        self.test_images= test_images
        self.test_labels= test_labels

    def evaluate_error_and_cost(self, filter_sizes, dense_sizes, alpha, l2_regul, dropout, num_epoch=0):


        filter_sizes= np.ndarray.astype(np.rint(np.power(2, filter_sizes)), dtype= int)
        dense_sizes= np.ndarray.astype(np.rint(np.power(2, dense_sizes)), dtype= int)
        alpha= np.power(10, alpha)
        l2_regul= np.power(10, l2_regul)
        # dropout= np.power(10, dropout)
        num_epoch= np.int(np.rint(np.power(10, num_epoch)))

        xt_eval= np.empty([1, len(filter_sizes)+ len(dense_sizes)+3])

        xt_eval[0, 0:len(filter_sizes)]= np.log2(filter_sizes[:])
        xt_eval[0,len(filter_sizes):len(filter_sizes)+ len(dense_sizes)]= np.log2(dense_sizes[:])
        xt_eval[0, len(filter_sizes)+ len(dense_sizes)]= np.log10(alpha)
        xt_eval[0, len(filter_sizes)+ len(dense_sizes)+1]= np.log10(l2_regul)
        xt_eval[0, len(filter_sizes)+ len(dense_sizes)+2]= dropout


        t1 = time.clock()
        model = Sequential()

        model.add(
            Conv2D(filter_sizes[0], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                   input_shape=(32, 32, 3), kernel_regularizer=l2(l2_regul)))
        model.add(BatchNormalization())

        model.add(Conv2D(filter_sizes[0], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_regul)))
        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))


        # model.add(Conv2D(filter_sizes[1], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                  kernel_regularizer=l2(l2_regul)))
        # model.add(BatchNormalization())
        #
        # model.add(Conv2D(filter_sizes[1], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                  kernel_regularizer=l2(l2_regul)))
        # model.add(BatchNormalization())
        #
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Dropout(0.3))


        # model.add(Conv2D(filter_sizes[2], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                  kernel_regularizer=l2(l2_regul)))
        # model.add(BatchNormalization())
        #
        # model.add(Conv2D(filter_sizes[2], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                  kernel_regularizer=l2(l2_regul)))
        # model.add(BatchNormalization())
        #
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Dropout(0.4))


        model.add(Flatten())
        model.add(Dense(dense_sizes[0], activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_regul)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(10, activation='softmax'))


        # compile model
        opt = Adam(lr=alpha)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        '''fit'''
        model.fit(self.train_images, self.train_labels, epochs= num_epoch)

        '''test set accuracy'''
        test_loss, test_acc = model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        t2= time.clock()

        return np.atleast_2d(test_loss), np.atleast_2d(t2- t1), xt_eval


class Keras_model_boston():

    def __init__(self):
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()
        self.X_train= X_train
        self.Y_train= Y_train
        self.X_test= X_test
        self.Y_test= Y_test


    def evaluate_error_and_cost(self, layer_sizes, alpha, l2_regul, num_epoch):

        layer_sizes = np.ndarray.astype(np.rint(np.power(2, layer_sizes)), dtype= int)
        alpha= np.power(10.0, alpha)
        l2_regul= np.power(10.0, l2_regul)
        num_epoch= np.int(np.rint(np.power(10, num_epoch)))
        print('layer_sizes:{}\nalpha:{}\nl2_regul:{}\nnum_epoch:{}'.format(layer_sizes, alpha, l2_regul, num_epoch))
        '''define model'''
        t1= time.clock()
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(13, )),
            keras.layers.Dense(layer_sizes[0], activation='relu',  kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[1], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[2], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[3], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(1)
        ])
        opt = keras.optimizers.Adam(learning_rate= alpha)
        '''compile model'''
        model.compile(optimizer= opt,  loss=tf.keras.losses.mean_squared_error,
                      metrics=['MeanSquaredError'])

        '''fit'''
        model.fit(self.X_train, self.Y_train, epochs= num_epoch)

        '''test set accuracy'''
        test_loss, test_mse = model.evaluate(self.X_test,  self.Y_test, verbose=2)
        print('\nTest mse:', test_mse)
        t2= time.clock()

        return np.atleast_2d(test_mse), np.atleast_2d(t2- t1), model


def uniformly_choose_from_domain(domain, num_samples):
    X= np.empty([num_samples, len(domain)])
    # Y= np.empty([num_samples, 1])

    for i in range(len(domain)):
        xi= np.random.uniform(domain[i][0], domain[i][1], (num_samples,1))
        X[:, i]= xi[:,0]
    return X

def initial_training_cifar(domain, num_samples, num_layer, num_dense, num_epoch= 1.7):

    X= uniformly_choose_from_domain(domain, num_samples)

    keras_model = Keras_model_cifar()

    Y= np.empty([X.shape[0], 1])
    Y_cost= np.empty([X.shape[0], 1])

    for i in range(X.shape[0]):
        xi= X[i, :].reshape(1,-1)
        z= num_layer+num_dense
        filter_sizes =xi[0, 0:num_layer]; dense_sizes = xi[0, num_layer:z];
        alpha = xi[0, z]; l2_regul = xi[0, z+1]; dropout = xi[0, z+2]

        Y[i, 0], Y_cost[i, 0], X[i, :]= keras_model.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul, dropout,
                                                                  num_epoch= num_epoch)

    return X, Y, Y_cost

def initial_training_fashion(domain, num_samples, num_layer, num_epoch= 1.7):

    X= uniformly_choose_from_domain(domain, num_samples)

    keras_model= Keras_model_fashion()

    Y= np.empty([X.shape[0], 1])
    Y_cost= np.empty([X.shape[0], 1])

    for i in range(X.shape[0]):
        xi= X[i, :].reshape(1,-1)
        layer_sizes =xi[0, 0:num_layer]; alpha = xi[0, num_layer]; l2_regul = xi[0, num_layer+1];
        #num_epoch= xi[0, num_layer+2]

        Y[i, 0], Y_cost[i, 0], X[i, :]= keras_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul, num_epoch)

    return X, Y, Y_cost


def get_cifar_domain(num_layer, num_dense):

    domain_filter_size_log2= [0.0, 8.0]; domain_dense_size_log2= [0.0, 8.0]; domain_alpha_log10= [-6.0, 0.0];
    domain_l2_regul = [0.0, 0.1]; domain_dropout= [0.0, 0.8]; domain_num_epoch_log10 = [0, 2]

    domain= []

    for i in range(num_layer):
        domain.append(domain_filter_size_log2)
    for k in range(num_dense):
        domain.append(domain_dense_size_log2)

    domain.append(domain_alpha_log10)
    domain.append(domain_l2_regul)
    domain.append(domain_dropout)
    # domain.append(domain_num_epoch_log10)
    return domain

def get_fashion_domain(num_layer):

    domain_layer_size_log2= [0.0, 8.0]; domain_alpha_log10= [-6.0, 0.0];
    domain_l2_regul = [-8.0, -1.0];  #domain_num_epoch_log10 = [0, 1.3]

    domain= []

    for i in range(num_layer):
        domain.append(domain_layer_size_log2)


    domain.append(domain_alpha_log10)
    domain.append(domain_l2_regul)
    #domain.append(domain_num_epoch_log10)

    return domain

def logistic_bjt(lower, upper):
    '''define logistic bijector'''
    low= tf.constant(lower, dtype= tf.float64); high= tf.constant(upper, dtype= tf.float64)
    affine = tfp.bijectors.AffineScalar(shift=low.numpy(), scale=(high.numpy() - low.numpy()))
    sigmoid = tfp.bijectors.Sigmoid()
    logistic = tfp.bijectors.Chain([affine, sigmoid])
    return logistic

def find_best_suited_gp_kernels(X, Y, Y_cost, noise, noise_cost):


    '''constraint values'''
    lower= 10**(-3); upper= 10**(6); #lengtscale and variance constarint
    lower_noise= 10**(-3); upper_noise= 10**(6); #noise constarint

    latent_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((X, np.log(Y_cost)), latent_kernel)
    '''limit latent cost kernel hyperparameters'''
    latent_cost_model.kernel.lengthscales= gp.Parameter(latent_cost_model.kernel.lengthscales.numpy(),transform=logistic_bjt(lower, upper))
    latent_cost_model.kernel.variance= gp.Parameter(latent_cost_model.kernel.variance.numpy(), transform= logistic_bjt(lower, upper))
    latent_cost_model.likelihood.variance= gp.Parameter(latent_cost_model.likelihood.variance.numpy(), transform= logistic_bjt(lower_noise, upper_noise))

    # latent_cost_model.likelihood.variance.assign(noise_cost)
    '''optimize cost model hyperparamters'''
    # gp.set_trainable(latent_cost_model.likelihood, False)
    opt_cost = gp.optimizers.Scipy()
    opt_cost.minimize(latent_cost_model.training_loss, latent_cost_model.trainable_variables, options=dict(maxiter=100))



    kernel= gp.kernels.RBF()
    model= gp.models.GPR((X, Y), kernel)
    '''limit objective kernel hyperparameters'''
    model.kernel.lengthscales= gp.Parameter(model.kernel.lengthscales.numpy(), transform= logistic_bjt(lower, upper))
    model.kernel.variance= gp.Parameter(model.kernel.variance.numpy(), transform= logistic_bjt(lower, upper))
    model.likelihood.variance= gp.Parameter(model.likelihood.variance.numpy(), transform= logistic_bjt(lower_noise, upper_noise))

    # model.likelihood.variance.assign(noise)


    '''optimize objective model hyperparamters'''
    opt_obj = gp.optimizers.Scipy()
    opt_obj.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

    print_summary(model)
    print_summary(latent_cost_model)

    return kernel, latent_kernel

def test_model_fashion():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    keras_model= Keras_model_fashion()
    domain_alpha= [1e-6, 1.0]
    domain_l2= [0, 0.1]
    domain_epochs= [1,100]
    layer_sizes= [32, 32, 32, 32, 32]; alpha_log= -4; l2_regul_log= -2; num_epoch_log= 1
    l2_regul= 10**(l2_regul_log); alpha= 10**(alpha_log); num_epoch= 10**(num_epoch_log)
    test_loss, time_cost= keras_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul, num_epoch)
    print('test_loss:{}, time_cost:{}'.format(test_loss, time_cost))

def test_model_cifar():
    #batch size
    # (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()


    filter_sizes, dense_sizes, alpha, l2_regul, num_epoch, dropout= np.array([32, 64, 128]), np.array([128]), 10**(-4), 0.01, 1, 0.1
    model_cifar= Keras_model_cifar()
    test_loss, time_cost= model_cifar.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul, dropout)

def test_boston_model():

    layer_sizes= [5, 5, 5, 5]; alpha_log= -4; l2_regul_log= -2; num_epoch_log= 1
    boston_model= Keras_model_boston()
    test_loss, time_cost, model= boston_model.evaluate_error_and_cost(layer_sizes, alpha_log, l2_regul_log, 2)
    index= 2
    x_test= (boston_model.X_test[index, :]).reshape(1,-1)
    y_test = (boston_model.Y_test[index])
    print(model.predict(x_test))
    print('mean squared error:{}, time_cost:{}'.format(test_loss, time_cost))
