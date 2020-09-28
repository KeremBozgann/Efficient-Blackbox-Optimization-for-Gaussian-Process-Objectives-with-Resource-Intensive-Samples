import numpy as np
import math
import matplotlib.pyplot as plt
norm= lambda x: 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)

x= 0
sample_num =1000
x_list= []
for i in range(sample_num):

    y= np.random.uniform(0, norm(x))
    x_lower= -math.sqrt(2*math.log(1/(math.sqrt(2*math.pi)*y)))
    x_upper= math.sqrt(2*math.log(1/(math.sqrt(2*math.pi)*y)))
    x_list.append(x)
    x= np.random.uniform(x_lower, x_upper)

X= np.linspace(-10,10, 1000)
plt.figure()
plt.hist(np.array(x_list),bins=50, density= 1/sample_num)
plt.plot(X, norm(X))