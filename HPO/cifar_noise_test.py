
from keras_model import Keras_model_cifar
from keras_model import Keras_model_fashion
from keras_model import get_cifar_domain
from keras_model import get_fashion_domain
from keras_model import uniformly_choose_from_domain
import numpy as np
import gpflow as gp
from gpflow.utilities import print_summary
import tensorflow_probability as tfp
import tensorflow as tf

def cifar_noise_test():

    num_layer=1 ; num_dense=1

    domain= get_cifar_domain(num_layer, num_dense)

    X = uniformly_choose_from_domain(domain, 10)

    keras_model = Keras_model_cifar()

    iterations = 10

    Y = np.empty([iterations * X.shape[1], 1])
    Y_cost = np.empty([iterations * X.shape[1], 1])

    y = np.empty([iterations, 1])
    y_cost = np.empty([iterations, 1])

    mean_list_obj = np.empty([X.shape[0], 1]);
    mean_list_cost = np.empty([X.shape[0], 1])
    std_list_obj = np.empty([X.shape[0], 1]);
    std_list_cost = np.empty([X.shape[0], 1])

    XX = np.empty([iterations * X.shape[0], X.shape[1]])

    for j in range(X.shape[0]):

        x = X[j, :].reshape(1, -1)
        print('x', x)
        for i in range(iterations):
            XX[j * iterations + i, :] = x[0, :]

            z = num_layer + num_dense
            filter_sizes = x[0, 0:num_layer];
            dense_sizes = x[0, num_layer:z];
            alpha = x[0, z];
            l2_regul = x[0, z + 1];
            dropout = x[0, z + 2]

            y[i, 0], y_cost[i, 0] = keras_model.evaluate_error_and_cost(filter_sizes, dense_sizes, alpha, l2_regul,
                                                                        dropout)

        print('mean_y:{}, std_y:{}'.format(np.mean(y), np.std(y)))
        print('mean_y_cost:{}, std_y_cost:{}'.format(np.mean(y_cost), np.std(y_cost)))
        mean_list_obj[j, 0] = np.mean(y);
        mean_list_cost[j, :] = np.mean(np.log(y_cost))
        std_list_obj[j, 0] = np.std(y);
        std_list_cost[j, :] = np.std(np.log(y_cost))

    kernel = gp.kernels.RBF()
    latent_cost_kernel = gp.kernels.RBF()

    model = gp.models.GPR((XX, Y), kernel)
    latent_cost_model = gp.models.GPR((X, np.log(Y_cost)), latent_cost_kernel)

    opt_cost = gp.optimizers.Scipy()
    opt_cost.minimize(latent_cost_model.training_loss, latent_cost_model.trainable_variables, options=dict(maxiter=100))

    opt = gp.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

    print_summary(model)
    print_summary(latent_cost_model)

def logistic_bjt(lower, upper):
    '''define logistic bijector'''
    low= tf.constant(lower, dtype= tf.float64); high= tf.constant(upper, dtype= tf.float64)
    affine = tfp.bijectors.AffineScalar(shift=low.numpy(), scale=(high.numpy() - low.numpy()))
    sigmoid = tfp.bijectors.Sigmoid()
    logistic = tfp.bijectors.Chain([affine, sigmoid])
    return logistic
def fashion_noise_test():

    num_layer=4

    domain= get_fashion_domain(num_layer)

    X = uniformly_choose_from_domain(domain, 5)

    keras_model = Keras_model_fashion()
    iterations = 5

    Y = np.empty([iterations*X.shape[0], 1])
    Y_cost = np.empty([iterations*X.shape[0], 1])

    y = np.empty([iterations, 1])
    y_cost = np.empty([iterations, 1])

    mean_list_obj= np.empty([X.shape[0], 1]); mean_list_cost= np.empty([X.shape[0], 1])
    std_list_obj= np.empty([X.shape[0], 1]); std_list_cost= np.empty([X.shape[0], 1])

    XX= np.empty([iterations*X.shape[0], X.shape[1]])

    for j in range(X.shape[0]):

        x= X[j, :].reshape(1,-1)
        print('x', x)
        for i in range(iterations):
            XX[j*iterations+i, :] = x[0,:]
            layer_sizes = x[0, 0:num_layer];
            alpha = x[0, num_layer];
            l2_regul = x[0,num_layer + 1];
            num_epoch= x[0,num_layer + 2];


            y[i, 0], y_cost[i, 0] = keras_model.evaluate_error_and_cost(layer_sizes, alpha, l2_regul, num_epoch)
            Y[j*iterations+i, 0]= y[i, 0]
            Y_cost[j*iterations+i, 0]= y_cost[i, 0]

        print('mean_y:{}, std_y:{}'.format(np.mean(y), np.std(y)))
        print('mean_y_cost:{}, std_y_cost:{}'.format(np.mean(y_cost), np.std(y_cost)))
        mean_list_obj[j, 0]= np.mean(y); mean_list_cost[j,:]= np.mean(np.log(y_cost))
        std_list_obj[j, 0]= np.std(y); std_list_cost[j,:]= np.std(np.log(y_cost))





    '''ard kernels'''
    kernel= gp.kernels.RBF(lengthscales=np.array([1]*X.shape[1]))
    latent_cost_kernel= gp.kernels.RBF(lengthscales=np.array([1]*X.shape[1]))


    '''constraint values'''
    lower= 10**(-3); upper= 10**(6); #lengtscale and variance constarint
    lower_noise= 10**(-6); upper_noise= 10**(6); #noise constarint

    '''define models with constraints'''
    model= gp.models.GPR((XX, Y), kernel)
    model.kernel.lengthscales= gp.Parameter(model.kernel.lengthscales.numpy(), transform= logistic_bjt(lower, upper))
    model.kernel.variance= gp.Parameter(model.kernel.variance.numpy(), transform= logistic_bjt(lower, upper))
    model.likelihood.variance= gp.Parameter(model.likelihood.variance.numpy(),transform= logistic_bjt(lower_noise, upper_noise))

    latent_cost_model= gp.models.GPR((XX, np.log(Y_cost)), latent_cost_kernel)
    latent_cost_model.kernel.lengthscales= gp.Parameter(latent_cost_model.kernel.lengthscales.numpy(),transform= logistic_bjt(lower, upper))
    latent_cost_model.kernel.variance= gp.Parameter(latent_cost_model.kernel.variance.numpy(), transform= logistic_bjt(lower, upper))
    latent_cost_model.likelihood.variance= gp.Parameter(latent_cost_model.likelihood.variance.numpy(),transform= logistic_bjt(lower_noise, upper_noise))


    opt_cost = gp.optimizers.Scipy()
    opt_cost.minimize(latent_cost_model.training_loss, latent_cost_model.trainable_variables, options=dict(maxiter=100))

    opt = gp.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

