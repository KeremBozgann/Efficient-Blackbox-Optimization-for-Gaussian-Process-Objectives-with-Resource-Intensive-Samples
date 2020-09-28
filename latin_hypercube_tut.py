
import numpy as np
import matplotlib.pyplot as plt

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