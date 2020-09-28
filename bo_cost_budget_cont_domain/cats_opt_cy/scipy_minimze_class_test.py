import numpy as np
from scipy.optimize import minimize, Bounds

class Optimize():


    def __init__(self):

        pass

    def function(self, x):
        print('function')
        self.z= x**2

        return -(self.z+ x**3)

    def gradient(self, x):
        print('gradient')
        return -(2*np.sqrt(self.z)+ 3*x**2)

    def minimize(self):
        fun= self.function
        grad= self.gradient
        x0= [1.0]
        domain= [[0.0, 2.0]]
        lower= [0.0]; upper= [2.0]
        b = Bounds(lb=lower, ub=upper)
        print(minimize(fun, x0, bounds= b, jac= grad))
