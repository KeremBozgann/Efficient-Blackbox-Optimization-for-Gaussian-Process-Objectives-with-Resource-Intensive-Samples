import numpy as np
import matplotlib.pyplot as plt
import gpflow as gp
import h5py
import tensorflow as tf

noise= 10**(-4)

with h5py.File('./datasets/1d/gp_sample_rbf_l_1.0_v_1.0.h5', 'r') as hf:
    X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

ind_rand= np.random.permutation(np.arange(X.shape[0]))[0:10]
Xt= X[ind_rand,:]; Yt= Y[ind_rand, :]
kernel= gp.kernels.RBF()
model= gp.models.GPR((Xt,Yt), kernel= kernel)
model.likelihood.variance.assign(noise)
mu, var= model.predict_f(X)

log_ent_reduct_list = np.empty([X.shape[0], 1])

for i in range(X.shape[0]):
    # x_psed is equal to x
    x_psed = X[i, :].reshape(1, -1);
    Xt_psed = np.append(Xt, x_psed, axis=0)
    temp1 = tf.matmul(tf.matmul(kernel.K(X, Xt), tf.linalg.inv(
        tf.add(kernel.K(Xt, Xt), tf.cast(noise * tf.eye(Xt.shape[0]), tf.float64)))), tf.transpose(kernel.K(X, Xt)))
    temp2 = tf.matmul(tf.matmul(kernel.K(X, Xt_psed), tf.linalg.inv(
        tf.add(kernel.K(Xt_psed, Xt_psed), tf.cast(noise * tf.eye(Xt_psed.shape[0]), tf.float64)), tf.float64)),
                      tf.transpose(kernel.K(X, Xt_psed)))


    log_ent_prior = np.log(1 / 2 * (2 * np.pi * np.e))*(Xt.shape[0]) + np.log(np.linalg.det(kernel.K(X, X) - temp1))
    log_ent_psed = np.log(1 / 2 * (2 * np.pi * np.e))*(Xt.shape[0]) + np.log(np.linalg.det(kernel.K(X, X) - temp2))
    log_entropy_reduct_x = log_ent_prior - log_ent_psed

    log_ent_reduct_list[i, 0] = (log_entropy_reduct_x)

index_max= np.argmax(log_ent_reduct_list); x_max= X[index_max,:].reshape(-1,1); y_max= Y[index_max,:].reshape(-1,1)

'''show the point that has results in maximum entropy reduction'''
plt.figure()
plt.scatter(X, np.zeros(X.shape[0],), color='black', marker='X')
plt.scatter(Xt,Yt, color='orange', marker='X')
plt.scatter(x_max,y_max, color='blue', marker='X')
plt.plot(X, mu, color= 'red')
plt.fill_between(X, mu+np.sqrt(var), mu-np.sqrt(var), color='ligthgrey', alpha=0.5)
plt.show()