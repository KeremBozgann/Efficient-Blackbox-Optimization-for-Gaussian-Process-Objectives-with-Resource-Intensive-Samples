import numpy as np
from scipy.stats import norm
import gpflow as gp

def EI_per_sec(x, sigma_x, u_x , f_best, cost_m_x):

    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama*fi_x+ norm.pdf(x))
    EI_x_per_sec= EI_x/cost_m_x

    return EI_x_per_sec

def EI(x, sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama*fi_x+ norm.pdf(x))

    return EI_x


kernel= gp.kernels.RBF()
model= gp.models.GPR(()kernel)
