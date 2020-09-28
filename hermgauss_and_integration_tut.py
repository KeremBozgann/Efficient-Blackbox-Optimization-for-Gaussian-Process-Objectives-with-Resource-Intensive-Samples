
'''integration'''
from scipy.integrate import quad
import numpy as np

from scipy.stats import norm

def integrand(x, a, b):

    return np.exp(-(x**2))*(a*x**2+b)

def integrand_herm(x, a, b):

    return (a*x**2+b)

a= 20; b= 10
true= quad(integrand, a= -np.inf, b=np.inf, args=(a, b))[0]

'''gaussian hermite app'''

X, W= np.polynomial.hermite.hermgauss(3)

sum= 0
for i in range(len(X)):
    xi= X[i]; wi= W[i]
    sum+= integrand_herm(xi, a, b)* wi

print('true integral:{}, gaussherm_app:{}'.format(true, sum))


'''integral of gaussian distribution weighted functions'''

def integrand_gauss(x, a, b):

    result= norm.pdf(x)*(a*x**4+b)
    #result_herm= np.sqrt(2)*(np.exp(-x**2)*(a*(np.sqrt(2))**1*x+b))

    return result

def integrand_gauss_herm(x, a, b):

    x= np.sqrt(2)*x
    result= 1/np.sqrt(np.pi)*(a*x**4+b)

    return result

a= -5; b=50
true= quad(integrand_gauss, a= -np.inf, b=np.inf, args=(a, b))[0]


'''gaussian hermite app'''

X, W= np.polynomial.hermite.hermgauss(3)

sum= 0
for i in range(len(X)):
    xi= X[i]; wi= W[i]
    sum+= integrand_gauss_herm(xi, a, b)* wi

print('true integral:{}, gaussherm_app:{}'.format(true, sum))


