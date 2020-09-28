import numpy as np
from scipy.stats import norm
import gpflow as gp
from scipy.optimize import minimize
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution


def EI(sigma_x, u_x, f_best):
    gama_x = (f_best - u_x) / sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x = sigma_x * (gama_x * fi_x + norm.pdf(gama_x))

    return EI_x


def EI_optimize(model, f_best, domain, X, sigma_X, u_X):
    def func(x):
        # print('x', x)

        x = x.reshape(1, -1)
        u_x, var_x = model.predict_f(x);
        sigma_x = np.sqrt(var_x)
        gama_x = (f_best - u_x) / sigma_x
        fi_x = norm.cdf(gama_x)
        EI_x = float(sigma_x * (gama_x * fi_x + norm.pdf(gama_x)))

        # print('EI_x', EI_x)
        return -EI_x

    # domain= [[-5,10], [0,15]]
    # x1= np.random.uniform(low= domain[0][0], high= domain[0][1])
    # x2= np.random.uniform(low= domain[1][0], high= domain[1][1])
    # print('f_best', f_best)

    EI_X = EI(sigma_X, u_X, f_best);
    index0 = int(np.argmax(EI_X, axis=0))
    x0 = X[index0, :]

    x_opt = minimize(func, x0, bounds=domain, method='L-BFGS-B')['x']

    # x_opt = shgo(func, bounds=domain)['x']
    # x_opt = dual_annealing(func, bounds=domain)['x']
    # x_opt = differential_evolution(func, bounds=domain)['x']

    x_opt = x_opt.reshape(1, -1)

    return x_opt


def EI_cool(sigma_x, u_x, f_best, u_cost, C, budget, budget_init):
    gama_x = (f_best - u_x) / sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x = sigma_x * (gama_x * fi_x + norm.pdf(gama_x))

    EI_cool = EI_x / (u_cost) ** ((budget - C) / (budget - budget_init))

    return EI_cool


def EI_pu(sigma_x, u_x, f_best, u_cost):
    gama_x = (f_best - u_x) / sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x = sigma_x * (gama_x * fi_x + norm.pdf(gama_x))

    EI_pu = EI_x / u_cost

    return EI_pu


def EI_pu_optimize(model, f_best, domain, X, sigma_X, u_X, latent_cost_model, u_cost_X):
    def func(x):
        # print('x', x)

        x = x.reshape(1, -1)
        u_x, var_x = model.predict_f(x);
        sigma_x = np.sqrt(var_x)
        gama_x = (f_best - u_x) / sigma_x
        fi_x = norm.cdf(gama_x)
        EI_x = float(sigma_x * (gama_x * fi_x + norm.pdf(gama_x)))

        u_cost_latent, var_cost_latent = latent_cost_model.predict_f(x);
        u_cost_latent = u_cost_latent.numpy()
        u_cost_x = np.exp(u_cost_latent)
        EI_pu_x = EI_x / u_cost_x

        # print('EI_pu_x', EI_pu_x)
        return -EI_pu_x

    # domain= [[-5,10], [0,15]]
    # x1= np.random.uniform(low= domain[0][0], high= domain[0][1])
    # x2= np.random.uniform(low= domain[1][0], high= domain[1][1])
    # print('f_best', f_best)

    EI_pu_X = EI_pu(sigma_X, u_X, f_best, u_cost_X)
    index0 = int(np.argmax(EI_pu_X, axis=0))
    x0 = X[index0, :]

    x_opt = minimize(func, x0, bounds=domain, method='L-BFGS-B')['x']

    # x_opt = shgo(func, bounds=domain)['x']
    # x_opt = dual_annealing(func, bounds=domain)['x']
    # x_opt = differential_evolution(func, bounds=domain)['x']

    x_opt = x_opt.reshape(1, -1)

    return x_opt

