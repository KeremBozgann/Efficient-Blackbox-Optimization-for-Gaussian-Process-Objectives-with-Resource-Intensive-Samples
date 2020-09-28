

import gpflow as gp
import numpy as np
f_best= -0.1
Xt= np.random.rand(3,2); Yt= np.random.rand(3,1); Z= np.random.normal(0.0, 1.0, (1,1))
x1= np.random.uniform(1.0, 2.0, (2,1))
kernel= gp.kernels.RBF()
model= gp.models.GPR((Xt, Yt), kernel)
latent_cost_model= gp.models.GPR((Xt, Yt), kernel)
noise= 10**(-3)

import pyximport; pyximport.install(language_level= 3)

import cython_test_imports
print(cython_test_imports.CATS_negative_improvement(x1, Z, model, latent_cost_model, Xt, Yt, f_best, kernel, noise))
