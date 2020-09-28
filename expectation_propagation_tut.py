import numpy as np
import math
import matplotlib.pyplot as plt

def normal_distribution_multiplication(u1, u2, sigma1, sigma2):

    u= (sigma1**2*u2+sigma2**2*u1)/(sigma1**2+sigma2**2)
    var= 1/(1/sigma1**2+1/sigma2**2)

    return u, var

def normal_distribution_division(u1, u2, sigma1, sigma2):

    #u1= (sigma2**2*u3+sigma3**2*u2)/(sigma2**2+sigma3**2)
    #var1 = 1/(1/sigma2**2+1/sigma3**2)
    sigma3=  np.sqrt((1/sigma1**2- 1/sigma2**2)**(-1))
    u3= (u1*(sigma2**2+sigma3**2)-sigma3**2*u2)/(sigma2**2)
    return u3, sigma3

norm= lambda x, u, sigma: 1/np.sqrt(2*np.pi*sigma)*np.exp(-1/(2*sigma**2)*(x-u)**2)



x1= np.linspace(-5,5,21)
x2= np.linspace(-5,5,21)
X1, X2 = np.meshgrid(x1, x2); X1, X2 = X1.flatten(), X2.flatten(); X1, X2 = X1.reshape(-1, 1), X2.reshape(-1, 1)
X = np.append(X1, X2, axis=1)

u1= 2.0; u2= -2.0; u3= -4.0; sigma1= 1.0; sigma2= 1.0; sigma3= 1.0
#
# p1= norm(x1,u1,sigma1); p2= norm(x1,u2,sigma2)
# p= p1*p2
#
uq1= 0.0; uq2= 0.0; uq3= 0.0; sigmaq1= 1.0; sigmaq2= 1.0; sigmaq3= 1.0; varq1=sigmaq1**2; varq2= sigmaq2**2; varq3= sigmaq3*2
# q1= norm(x1,uq1,sigmaq1); q2= norm(x1,uq2,sigmaq2)
# Q= q1*q2

num_iter=5
for i in range(num_iter):
    '''step1'''
    #cavity= q2*q3; h= p1*cavity; match h and Q; q1= Q/cavity
    u_cav, var_cav= normal_distribution_multiplication(uq2, uq3, sigmaq2, sigmaq3); sigma_cav= np.sqrt(var_cav);
    u_h, var_h= normal_distribution_multiplication(u1, u_cav, sigma1, sigma_cav )

    uQ= u_h.copy(); varQ= var_h.copy(); sigmaQ = np.sqrt(varQ)

    uq1, varq1= normal_distribution_division(uQ, u_cav, sigmaQ, sigma_cav); sigmaq1= np.sqrt(varq1)

    '''step2'''
    #cavity= q1*q3; h= p2*cavity; match h and Q; q2= Q/cavity
    u_cav, var_cav= normal_distribution_multiplication(uq1, uq3, sigmaq1, sigmaq3); sigma_cav= np.sqrt(var_cav);
    u_h, var_h= normal_distribution_multiplication(u2, u_cav, sigma2, sigma_cav )

    uQ= u_h.copy(); varQ= var_h.copy(); sigmaQ = np.sqrt(varQ)

    uq2, varq2= normal_distribution_division(uQ, u_cav, sigmaQ, sigma_cav); sigmaq2= np.sqrt(varq2)

    '''step2'''
    #cavity= q1*q2; h= p3*cavity; match h and Q; q3= Q/cavity
    u_cav, var_cav= normal_distribution_multiplication(uq1, uq2, sigma1, sigma2); sigma_cav= np.sqrt(var_cav);
    u_h, var_h= normal_distribution_multiplication(u3, u_cav, sigma3, sigma_cav )

    uQ= u_h.copy(); varQ= var_h.copy(); sigmaQ = np.sqrt(varQ)

    uq3, varq3= normal_distribution_division(uQ, u_cav, sigmaQ, sigma_cav); sigmaq3= np.sqrt(varq3)


