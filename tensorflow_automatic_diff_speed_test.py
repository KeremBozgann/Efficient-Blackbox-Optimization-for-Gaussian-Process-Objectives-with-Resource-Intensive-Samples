
import tensorflow as tf
import numpy as np
import time

def fun(x):
    a= 1/x
    y= x**2+ x**3+ x**4+a

    return y

def fun_deriv(x):

    dy_dx= 2*x+ 3*x**2+ 4*x**3 - 1/x**2

    return dy_dx

x= tf.Variable(2.0)
with tf.GradientTape(persistent= True) as g:

    g.watch(x)
    dy_dx_tens= fun_deriv(x)

'''tensorflow gradient tape evaluation time'''
t1_tens=  time.clock()
g.gradient(dy_dx_tens, x)
t2_tens=  time.clock()
print('time tens:{}'.format(t2_tens- t1_tens))

'''gradient evaluation time using code'''
t1_code=  time.clock()
dy_dx= fun_deriv(x)
t2_code=  time.clock()
print('time code:{}'.format(t2_code- t1_code))
