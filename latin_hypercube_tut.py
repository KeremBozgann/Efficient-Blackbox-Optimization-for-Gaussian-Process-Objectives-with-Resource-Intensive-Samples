
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import *
from util import

def latin_hypercube_2d_uniform(n):

    lower_limits= np.arange(float(0), float(n))/n
    upper_limits= np.arange(1.0, n+1)/(n)

    points= np.random.uniform(low= lower_limits, high= upper_limits, size= [2,n]).T
    np.random.shuffle(points[:,1])

    return points

points= latin_hypercube_2d_uniform(10)

plt.figure(figsize= (5,5))
plt.xlim([0,1]); plt.ylim([0,1])
plt.scatter(points[:,0], points[:,1])
plt.show()

def scale(domain, X):

    for i in range(len(domain)):
        lowi= domain[i][0]
        highi= domain[i][1]
        middle=( highi+lowi)/2
        X[:,i]= (X[:,i]-0.5)*(highi- lowi)+middle

    return X

def test_pydoe():


    X = lhs(2, samples=1000, criterion='maximin')
    domain= [[0,15], [-5, 3]]
    X_rand = np.random.uniform(low= [domain[0][0], domain[1][0]], high= [domain[0][1],domain[1][1]], size = (1000, 2))
    X= scale(domain, X)
    plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], color= 'green')
    plt.scatter(X_rand[:, 0], X_rand[:, 1], color= 'red')
    plt.show()
