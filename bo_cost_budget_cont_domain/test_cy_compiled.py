import numpy as np
import gpflow as gp
import time

import cats_optimize_cy
from cats_optimize import *

# '''get_posterior_covariance_and_derivative_1q(x, x1, Xt, kernel, noise)'''
# x= np.random.rand(1,2)
# x1= np.random.rand(1,2)
# Xt= np.random.rand(3,2)
# kernel= gp.kernels.RBF()
# noise= 10**(-3)
#
# #compare py and cy times
# t1_cy= time.clock()
# cats_optimize_cy.get_posterior_covariance_and_derivative_1q(x, x1, Xt, kernel, noise)
# t2_cy= time.clock()
#
# t1_py= time.clock()
# get_posterior_covariance_and_derivative_1q(x, x1, Xt, kernel, noise)
# t2_py= time.clock()
#
# print('py code time:{}, cy code time:{}'.format((t2_py- t1_py),(t2_cy - t1_cy)))

'''get_posterior_covariance_and_derivative_1q(x, x1, Xt, kernel, noise)'''
x= np.random.rand(1,2)
x1= np.random.rand(1,2)
x2= np.random.rand(1,2)
Xt= np.random.rand(3,2)
Yt= np.random.rand(3,1)
K0_x1_x1=  np.random.rand(1,1)
dK0_x1_dx1=  np.random.rand(2,1)
D=2
L0_x1= np.random.rand(1,1)
L0_x1_inv= np.random.rand(1,1)
L0_x1_inv_T= np.random.rand(1,1)

kernel= gp.kernels.RBF()
noise= 10**(-3)

#compare py and cy times
t1_cy= time.clock()
print(cats_optimize_cy.get_dsigma0_dx1_q1(x2, x1, Xt, kernel, noise, dK0_x1_dx1, D, L0_x1, L0_x1_inv, L0_x1_inv_T))
t2_cy= time.clock()

t1_py= time.clock()
print(get_dsigma0_dx1_q1(x2, x1, Xt, kernel, noise, K0_x1_x1,  dK0_x1_dx1, D, Yt, L0_x1, L0_x1_inv, L0_x1_inv_T))
t2_py= time.clock()

print('py code time:{}, cy code time:{}'.format((t2_py- t1_py),(t2_cy - t1_cy)))
