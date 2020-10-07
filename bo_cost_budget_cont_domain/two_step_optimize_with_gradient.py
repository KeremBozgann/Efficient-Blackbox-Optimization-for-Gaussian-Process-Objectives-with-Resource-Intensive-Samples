
import numpy as np
from gp_gradients import *
from scipy.optimize import minimize
from scipy.optimize import Bounds


def two_opt_EI_optimize_herm(f_best, Xt, Yt, model, domain, noise, kernel, num_inner_opt_restarts,
                                                num_outer_opt_restarts, D, num_iter_max, grid_opt_in, grid_opt_out):


    def two_step_negative_improvement(x1, Z):

        x1 = x1.reshape(1,-1)
        # print('x1 shape:{}'.format(x1.shape))
        '''observation at x1'''
        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        u0_x1, var0_x1 = model.predict_f(x1);
        u0_x1 = u0_x1.numpy();
        var0_x1 = var0_x1.numpy()
        sigma0_x1 = np.sqrt(var0_x1)

        L0_x1=  np.linalg.cholesky(var0_x1)
        y1= u0_x1+ np.matmul(L0_x1,Z_conv)
        '''update Xt and Yt according to outcome of Z'''
        Xt1= np.append(Xt, x1, axis=0)
        Yt1= np.append(Yt, y1, axis=0)

        '''first two step improvement term'''
        i0_x1= np.maximum(f_best- float(np.min(Yt1, axis=0)), 0)

        if num_inner_opt_restarts>0:
            '''inner loop optimization with random restarts'''

            x2_opt_list_sci = [];
            opt_value_list_sci = [];
            grad_list_sci = [];
            x20_list_sci = []
            lower = [domain[i][0] for i in range(len(domain))];
            upper = [domain[i][1] for i in range(len(domain))]

            for i in range(num_inner_opt_restarts):
                # print('iter_inner:{}'.format(i))
                x20_sci = np.random.uniform(lower, upper, (1, D))
                x2_cand_sci, cand_value_sci, x2_cand_grad_sci, result = EI1_x2_optimize(x20_sci, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt,
                                                                            noise, model, f_best, domain)

                x2_cand_sci = x2_cand_sci.reshape(1, -1)
                x2_opt_list_sci.append(x2_cand_sci);
                opt_value_list_sci.append(cand_value_sci);
                grad_list_sci.append(x2_cand_grad_sci)
                x20_list_sci.append(x20_sci)

            index_opt_sci = int(np.argmax(np.array(opt_value_list_sci)))

            # global x2_opt

            x2_opt_sci = x2_opt_list[index_opt_sci]
            x2_opt_sci = x2_opt_sci.reshape(1, -1)
            x2_opt_value_sci = opt_value_list_sci[index_opt_sci]
            x2_opt_grad_sci = grad_list_sci[index_opt_sci]
            x20_opt_sci = x20_list_sci[index_opt_sci]

        '''inner optimization with grid'''
        if D == 1 and grid_opt_in:
            disc = 101
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            X2_grid = x1_grid.reshape(-1, 1)

            x2_opt_grid, x2_opt_value_grid= EI1_x2_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1, Z,
                                                                 Xt, Yt, noise, model, f_best)


        if D == 2 and grid_opt_in:
            disc = 21
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
            x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
            X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

            X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
            X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
            X2_grid = np.append(X1_flat, X2_flat, axis=1)

            x2_opt_grid, x2_opt_value_grid = EI1_x2_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1, Z,
                                                                  Xt, Yt, noise, model, f_best)

        '''compare grid with sci optimum'''

        if (D == 2 or D == 1) and grid_opt_in and (num_inner_opt_restarts>0):
            if x2_opt_value_grid > x2_opt_value_sci:
                x2_opt = x2_opt_grid;
                x2_opt_value = x2_opt_value_grid

            else:
                x2_opt = x2_opt_sci;
                x2_opt_value = x2_opt_value_sci

        elif num_inner_opt_restarts>0:
            x2_opt = x2_opt_sci;
            x2_opt_value = x2_opt_value_sci

        else:
            x2_opt= x2_opt_grid
            x2_opt_value= x2_opt_value_grid

        '''second two step improvement term'''
        ei1_x2= x2_opt_value
        ei1_x2= float(ei1_x2)

        two_opt= i0_x1+ ei1_x2

        if type(two_opt)!= float:

            two_opt= two_opt.flatten()
        # print('two_opt:{}'.format(two_opt))
        return -two_opt, x2_opt

    def two_step_negative_gradient(x1, Z, x2_opt):

        x1 = x1.reshape(1,-1)

        '''find cholesky gradient (L0_x1) and du0_x1_dx1'''
        du0_x1_dx1, dvar0_x1_dx1, K0_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

        u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
        sigma0_x1= np.sqrt(var0_x1)

        L0_x1= np.linalg.cholesky(var0_x1)

        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        if f_best- (u0_x1+ np.matmul(L0_x1, Z_conv))<0:

            term_grad1= 0
        else:

            # _, dK0_x1_dx1, K0_x1_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

            L0_x1 = np.linalg.cholesky(var0_x1); L0_x1_inv = np.linalg.inv(L0_x1);
            L0_x1_inv_T = np.transpose(L0_x1_inv)

            dL0_x1_dx1= np.empty([D,1])
            for i in range(D):
                dvar0_x1_dx1i= dvar0_x1_dx1[i,:].reshape(1,1)
                Ai= np.matmul(L0_x1_inv, np.matmul(dvar0_x1_dx1i, L0_x1_inv_T))
                temp= funfi(Ai)
                dL0_x1_dx1i = np.matmul(L0_x1, temp)
                dL0_x1_dx1[i,0]= dL0_x1_dx1i

            # term_grad1= -df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)
            term_grad1= -du0_x1_dx1- np.matmul(dL0_x1_dx1, Z_conv)

        # global x2_opt

        '''gradient of lookahead term'''
        u0_x2_opt, var0_x2_opt= model.predict_f(x2_opt); u0_x2_opt= u0_x2_opt.numpy(); var0_x2_opt= var0_x2_opt.numpy()
        sigma0_x1= np.sqrt(var0_x1)


        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        y1= u0_x1+ np.matmul(L0_x1, Z_conv)

        f_best1= np.minimum(f_best, float(y1))

        EI1_x2_grad_x1= dEI1_x2_dx1(f_best, f_best1, kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x1, u0_x2_opt,
                    var0_x2_opt)

        # print('term_grad1:{}, EI1_x2_grad_x1:{}'.format(term_grad1, EI1_x2_grad_x1))
        two_step_grad= term_grad1+ EI1_x2_grad_x1
        two_step_grad= two_step_grad.flatten()
        # print('two_step_grad:{}'.format(two_step_grad))
        return -two_step_grad


    monte_carlo_samples= 100
    Z_list = np.random.normal(0.0, 1.0, (1, monte_carlo_samples))


    def monte_carlo_average_two_step_improvement(x1):

        avg_negative_improvement= 0

        global x2_opt_list_mont
        x2_opt_list_mont= np.zeros([Z_list.shape[1], D])

        for k in range(Z_list.shape[1]):

            z= Z_list[:, k]
            negative_improvementk, x2_opt = two_step_negative_improvement(x1,z)
            avg_negative_improvement+= negative_improvementk/monte_carlo_samples

            x2_opt_list_mont[k, :]= x2_opt[0, :]


        return avg_negative_improvement

    def monte_carlo_average_two_step_gradient(x1):

        avg_negative_gradient = 0

        global x2_opt_list_mont

        for k in range(Z_list.shape[1]):

            x2_opt= x2_opt_list_mont[k, :].reshape(1,-1)

            z = Z_list[:, k]
            negative_improvementk = two_step_negative_gradient(x1, z, x2_opt)
            avg_negative_gradient += negative_improvementk / monte_carlo_samples

        return avg_negative_gradient

    Z, W= np.polynomial.hermite.hermgauss(20)

    def hermite_gaussian_app_two_step_improvement(x1):

        x1 = x1.reshape(1, -1)

        negative_improvement= 0

        global x2_opt_list
        x2_opt_list= np.zeros([len(Z), D])

        for k in range(len(Z)):
            # print('k:{}'.format(k))
            z= Z[k]; w= W[k]
            z*= np.sqrt(2)
            negative_improvementk, x2_opt =  two_step_negative_improvement(x1,z)
            negative_improvementk*= 1/np.sqrt(np.pi)*w
            negative_improvement+= negative_improvementk

            x2_opt_list[k, :]= x2_opt[0, :]

        return negative_improvement

    def hermite_gaussian_app_two_step_gradient(x1):

        x1 = x1.reshape(1, -1)

        negative_gradient = 0

        global x2_opt_list

        for k in range(len(Z)):

            x2_opt= x2_opt_list[k, :].reshape(1,-1)
            z= Z[k]; w= W[k]
            z *= np.sqrt(2)

            negative_improvementk = two_step_negative_gradient(x1,z, x2_opt)
            negative_improvementk *= 1/np.sqrt(np.pi)*w
            negative_gradient += negative_improvementk

        return negative_gradient

    if num_outer_opt_restarts > 0:

        lower= []; upper= []
        for i in range(len(domain)):
            lower.append(domain[i][0])
            upper.append(domain[i][1])
        b= Bounds(lb= lower, ub= upper )

        x1_opt_list_sci = [];
        opt_value_list_x1_sci = [];
        grad_list_x1_sci = [];
        x10_list_sci = []
        lower = [domain[i][0] for i in range(len(domain))]; upper = [domain[i][1] for i in range(len(domain))]


        for i in range(num_outer_opt_restarts):

            print('iter_outer:{}'.format(i))
            x10= np.random.uniform(lower, upper, (1,D))

            # herm_gauss_imp=hermite_gaussian_app_two_step_improvement(x10)
            # herm_gauss_grad= hermite_gaussian_app_two_step_gradient(x10)
            # monte_imp= monte_carlo_average_two_step_improvement(x10)
            # monte_grad= monte_carlo_average_two_step_gradient(x10)
            #
            # print('herm_app_two_step:{}, monte_app_two_step:{}, monte_carlo_samples:{}'.
            #       format(herm_gauss_imp, monte_imp, monte_carlo_samples))
            # print('herm_app_two_step_grad:{}, monte_app_two_step_grad:{}'.
            #       format(herm_gauss_grad, monte_grad))

            # print('x10 shape:{}'.format(x10.shape))
            result= (minimize(hermite_gaussian_app_two_step_improvement, x10, bounds=b,  method='L-BFGS-B',
                              jac=hermite_gaussian_app_two_step_gradient, options= {'maxiter': num_iter_max}))
            x1_cand_sci= result['x']; cand_value_x1_sci= -result['fun']; x1_cand_grad_sci= -result['jac']
            cand_value_x1_scu= cand_value_x1_sci.reshape(1,-1); x1_cand_grad_sci= x1_cand_grad_sci.reshape(1,-1)
            x1_cand_sci = x1_cand_sci.reshape(1, -1)

            x1_opt_list_sci.append(x1_cand_sci);
            opt_value_list_x1_sci.append(cand_value_x1_sci);
            grad_list_x1_sci.append(x1_cand_grad_sci)
            x10_list_sci.append(x10)

        index_opt_x1_sci = int(np.argmax(np.array(opt_value_list_x1_sci)))
        x1_opt_sci = x1_opt_list_sci[index_opt_x1_sci]
        x1_opt_sci= x1_opt_sci.reshape(1,-1)
        x1_opt_value_sci = opt_value_list_x1_sci[index_opt_x1_sci]
        x1_opt_grad_sci = grad_list_x1_sci[index_opt_x1_sci]
        x10_opt_sci = x10_list_sci[index_opt_x1_sci]

    if grid_opt_out== True:
        if D==1:

            disc = 101
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            X_grid = x1_grid.reshape(-1, 1)

            x1_value_list_grid= np.zeros([disc, 1])

            for j in range(disc):
                x1_cand= X_grid[j,:].reshape(1,-1)
                x1_cand_value= -hermite_gaussian_app_two_step_improvement(x1_cand);
                x1_cand_value= x1_cand_value.reshape(1,-1)
                x1_value_list_grid[j, :]= x1_cand_value[0,:]

            index_max_grid = int(np.argmax(x1_value_list_grid, axis=0))
            x_opt_value_grid =x1_value_list_grid[index_max_grid, :].reshape(1, -1)
            x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)

        if D == 2 :
            disc = 21
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
            x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
            X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

            X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
            X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
            X_grid = np.append(X1_flat, X2_flat, axis=1)

            x1_value_list_grid = np.zeros([disc**2, 1])

            for j in range(disc**2):
                x1_cand = X_grid[j, :].reshape(1, -1)
                x1_cand_value = -hermite_gaussian_app_two_step_improvement(x1_cand);
                x1_cand_value = x1_cand_value.reshape(1, -1)
                x1_value_list_grid[j, :] = x1_cand_value[0, :]

            index_max_grid = int(np.argmax(x1_value_list_grid, axis=0))
            x_opt_value_grid =x1_value_list_grid[index_max_grid, :].reshape(1, -1)
            x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)


    if (D == 2 or D == 1) and grid_opt_out and (num_outer_opt_restarts>0):
        if x_opt_value_grid > x_opt_value_grid:
            x_opt = x_opt_grid;
            x_opt_value = x_opt_value_grid


        else:
            x_opt = x1_opt_sci;
            x_opt_value = x1_opt_value_sci

        return x_opt, x_opt_value, x1_value_list_grid, X1_grid

    elif (not grid_opt_out) and (num_outer_opt_restarts>0):
        x_opt = x1_opt_sci;
        x_opt_value = x1_opt_value_sci

        return x_opt, x_opt_value, None, None

    elif grid_opt_out and (not num_outer_opt_restarts>0):
        x_opt= x_opt_grid
        x_opt_value= x_opt_value_grid

        return x_opt, x_opt_value, x1_value_list_grid, X_grid


    # return x1_opt_sci, x1_opt_value_sci,  x1_opt_grad_sci, result
    # return herm_gauss_imp, monte_imp, herm_gauss_grad, monte_grad

import sys
sys.path.append('../functions')

from sine import *

def test_hermite_gauss_approximation():

    domain= [[-2, 2]]; random_restarts= 2; D=1; noise= 10**(-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    x_opt_list= np.zeros([random_restarts, D])
    x_opt_value_list= np.zeros([random_restarts, 1])
    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= sin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 10
    num_outer_opt_restarts= 1

    two_opt_EI_optimize_herm(f_best, Xt, Yt, model, domain, noise, kernel, num_inner_opt_restarts,
                             num_outer_opt_restarts, D)


def test_hermite_gauss_approximation_2():

    domain= [[-2, 2]]; random_restarts= 2; D=1; noise= 10**(-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    x_opt_list= np.zeros([random_restarts, D])
    x_opt_value_list= np.zeros([random_restarts, 1])
    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= sin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    x2= np.random.uniform(lower, upper, (1,D))

    def improvement_temp(x1, Z):

        if type(Z) != np.ndarray:
            Z_conv = np.atleast_2d(Z)
        else:
            Z_conv= Z.copy()

        u0_x1, var0_x1 = model.predict_f(x1);
        u0_x1 = u0_x1.numpy();
        var0_x1 = var0_x1.numpy()
        sigma0_x1 = np.sqrt(var0_x1)

        L0_x1=  np.linalg.cholesky(var0_x1)
        y1= u0_x1+ np.matmul(L0_x1,Z_conv)
        '''update Xt and Yt according to outcome of Z'''
        Xt1= np.append(Xt, x1, axis=0)
        Yt1= np.append(Yt, y1, axis=0)

        '''first two step improvement term'''
        i0_x1= np.maximum(f_best- float(np.min(Yt1, axis=0)), 0)

        f_best1= float(np.amin(Yt1, axis=0))

        def EI_x2(x2):

            # print('x2 shape before reshaping', x2.shape)
            x2 = x2.reshape(1, -1)

            # print('x2 shape in obj', x2.shape)
            u0_x2, var0_x2 = model.predict_f(x2);
            u0_x2 = u0_x2.numpy();
            var0_x2 = var0_x2.numpy()

            u1_x2, var1_x2 = one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise)

            sigma1_x2 = np.sqrt(var1_x2)

            EI1_x2 = EI(sigma1_x2, u1_x2, f_best1)

            # print('EI1 shape before reshaping', EI1_x2.shape)
            EI1_x2 = EI1_x2.reshape(1, -1)

            EI1_x2 = EI1_x2.flatten()
            return EI1_x2

        x2_val= EI_x2(x2)


        '''second two step improvement term'''
        ei1_x2= x2_val
        ei1_x2= float(ei1_x2)

        two_opt= i0_x1+ ei1_x2

        if type(two_opt)!= float:

            two_opt= two_opt.flatten()
        # print('two_opt:{}'.format(two_opt))
        return -two_opt


    monte_carlo_samples= 1000
    Z_list = np.random.normal(0.0, 1.0, (1, monte_carlo_samples))

    def monte_imp_temp(x1):

        avg_negative_improvement = 0

        for k in range(Z_list.shape[1]):
            z = Z_list[:, k]
            negative_improvementk= improvement_temp(x1, z)
            avg_negative_improvement += negative_improvementk / monte_carlo_samples

        return avg_negative_improvement

    Z, W = np.polynomial.hermite.hermgauss(20)

    def herm_imp_temp(x1):

        negative_improvement = 0

        for k in range(len(Z)):
            z = Z[k]; w = W[k]

            z *= np.sqrt(2)
            negative_improvementk= improvement_temp(x1, z)
            negative_improvementk *= 1 / np.sqrt(np.pi)*w
            negative_improvement += negative_improvementk


        return negative_improvement

    x1= np.random.uniform(lower, upper, (1, D))
    mont_imp= monte_imp_temp(x1)
    herm_imp= herm_imp_temp(x1)

    print('herm_imp:{}, mont_imp:{}'.format(herm_imp, mont_imp))


def two_step_negative_improvement_only(x1, Z, model, f_best, num_inner_opt_restarts, domain, Xt, Yt,
                                       noise, kernel, grid_opt_in, D, with_for):

    x1 = x1.reshape(1, -1)
    # print('x1 shape:{}'.format(x1.shape))
    '''observation at x1'''
    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv = Z.copy()

    u0_x1, var0_x1 = model.predict_f(x1);
    u0_x1 = u0_x1.numpy();
    var0_x1 = var0_x1.numpy()
    sigma0_x1 = np.sqrt(var0_x1)

    L0_x1 = np.linalg.cholesky(var0_x1)
    y1 = u0_x1 + np.matmul(L0_x1, Z_conv)
    '''update Xt and Yt according to outcome of Z'''
    Xt1 = np.append(Xt, x1, axis=0)
    Yt1 = np.append(Yt, y1, axis=0)

    '''first two step improvement term'''
    i0_x1 = np.maximum(f_best - float(np.min(Yt1, axis=0)), 0)

    if num_inner_opt_restarts > 0:
        '''inner loop optimization with random restarts'''

        x2_opt_list_sci = [];
        opt_value_list_sci = [];
        grad_list_sci = [];
        x20_list_sci = []
        lower = [domain[i][0] for i in range(len(domain))];
        upper = [domain[i][1] for i in range(len(domain))]

        for i in range(num_inner_opt_restarts):
            # print('iter_inner:{}'.format(i))
            x20_sci = np.random.uniform(lower, upper, (1, D))
            x2_cand_sci, cand_value_sci, x2_cand_grad_sci, result = EI1_x2_optimize(x20_sci, u0_x1, var0_x1, kernel, x1,
                                                                                    Z, Xt, Yt,
                                                                                    noise, model, f_best, domain)

            x2_cand_sci = x2_cand_sci.reshape(1, -1)
            x2_opt_list_sci.append(x2_cand_sci);
            opt_value_list_sci.append(cand_value_sci);
            grad_list_sci.append(x2_cand_grad_sci)
            x20_list_sci.append(x20_sci)

        index_opt_sci = int(np.argmax(np.array(opt_value_list_sci)))

        # global x2_opt

        x2_opt_sci = x2_opt_list_sci[index_opt_sci]
        x2_opt_sci = x2_opt_sci.reshape(1, -1)
        x2_opt_value_sci = opt_value_list_sci[index_opt_sci]
        x2_opt_grad_sci = grad_list_sci[index_opt_sci]
        x20_opt_sci = x20_list_sci[index_opt_sci]

    '''inner optimization with grid'''
    if D == 1 and grid_opt_in:
        disc = 11
        x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
        X2_grid = x1_grid.reshape(-1, 1)

        if not with_for:
            x2_opt_grid, x2_opt_value_grid = EI1_x2_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1, Z,
                                                                  Xt, Yt, noise, model, f_best)
        else:
            x2_opt_grid, x2_opt_value_grid= EI1_x2_optimize_grid_with_for(X2_grid, u0_x1, var0_x1, kernel, x1, Z,
                                                                          Xt, Yt, noise, model, f_best)


    if D == 2 and grid_opt_in:
        disc = 21
        x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
        x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
        x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
        X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

        X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
        X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
        X2_grid = np.append(X1_flat, X2_flat, axis=1)

        if not with_for:
            x2_opt_grid, x2_opt_value_grid = EI1_x2_optimize_grid(X2_grid, u0_x1, var0_x1, kernel, x1, Z,
                                                                  Xt, Yt, noise, model, f_best)
        else:
            x2_opt_grid, x2_opt_value_grid= EI1_x2_optimize_grid_with_for(X2_grid, u0_x1, var0_x1, kernel, x1, Z,
                                                                          Xt, Yt, noise, model, f_best)

    '''compare grid with sci optimum'''

    if (D == 2 or D == 1) and grid_opt_in and (num_inner_opt_restarts > 0):
        if x2_opt_value_grid > x2_opt_value_sci:
            x2_opt = x2_opt_grid;
            x2_opt_value = x2_opt_value_grid

        else:
            x2_opt = x2_opt_sci;
            x2_opt_value = x2_opt_value_sci

    elif num_inner_opt_restarts > 0:
        x2_opt = x2_opt_sci;
        x2_opt_value = x2_opt_value_sci

    else:
        x2_opt = x2_opt_grid
        x2_opt_value = x2_opt_value_grid

    '''second two step improvement term'''
    ei1_x2 = x2_opt_value
    ei1_x2 = float(ei1_x2)

    two_opt = i0_x1 + ei1_x2

    if type(two_opt) != float:
        two_opt = two_opt.flatten()
    # print('two_opt:{}'.format(two_opt))
    return -two_opt, x2_opt, -i0_x1, -ei1_x2



def two_step_negative_gradient_only(x1, Z, x2_opt, kernel, Xt, Yt, noise, model, f_best, D):


    x1 = x1.reshape(1, -1)

    '''find cholesky gradient (L0_x1) and du0_x1_dx1'''
    du0_x1_dx1, dvar0_x1_dx1, K0_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

    u0_x1, var0_x1 = model.predict_f(x1);
    u0_x1 = u0_x1.numpy();
    var0_x1 = var0_x1.numpy()
    sigma0_x1 = np.sqrt(var0_x1)

    L0_x1 = np.linalg.cholesky(var0_x1)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv = Z.copy()

    if f_best - (u0_x1 + np.matmul(L0_x1, Z_conv)) < 0:

        term_grad1 = 0
    else:

        # _, dK0_x1_dx1, K0_x1_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

        L0_x1 = np.linalg.cholesky(var0_x1);
        L0_x1_inv = np.linalg.inv(L0_x1);
        L0_x1_inv_T = np.transpose(L0_x1_inv)

        dL0_x1_dx1 = np.empty([D, 1])
        for i in range(D):
            dvar0_x1_dx1i = dvar0_x1_dx1[i, :].reshape(1, 1)
            Ai = np.matmul(L0_x1_inv, np.matmul(dvar0_x1_dx1i, L0_x1_inv_T))
            temp = funfi(Ai)
            dL0_x1_dx1i = np.matmul(L0_x1, temp)
            dL0_x1_dx1[i, 0] = dL0_x1_dx1i

        # term_grad1= -df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)
        term_grad1 = -du0_x1_dx1 - np.matmul(dL0_x1_dx1, Z_conv)

    print('term_grad1:{}'.format(term_grad1))

    # global x2_opt

    '''gradient of lookahead term'''
    u0_x2_opt, var0_x2_opt = model.predict_f(x2_opt);
    u0_x2_opt = u0_x2_opt.numpy();
    var0_x2_opt = var0_x2_opt.numpy()
    sigma0_x2_opt = np.sqrt(var0_x2_opt)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv = Z.copy()

    y1 = u0_x1 + np.matmul(L0_x1, Z_conv)

    f_best1 = np.minimum(f_best, float(y1))

    EI1_x2_grad_x1 = dEI1_x2_dx1(f_best, f_best1, kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x1, u0_x2_opt,
                                 var0_x2_opt)

    # print('term_grad1:{}, EI1_x2_grad_x1:{}'.format(term_grad1, EI1_x2_grad_x1))
    two_step_grad = term_grad1 + EI1_x2_grad_x1
    two_step_grad = two_step_grad.flatten()
    # print('two_step_grad:{}'.format(two_step_grad))
    return -two_step_grad, -term_grad1, -EI1_x2_grad_x1


def hermite_gaussian_app_two_step_improvement_only(x1, Z, W, model, f_best, num_inner_opt_restarts,domain, Xt, Yt,
                                                   noise, kernel, grid_opt_in, D, with_for ):

    x1 = x1.reshape(1, -1)

    negative_improvement= 0

    x2_opt_list= np.zeros([len(Z), D])

    term1_neg_avg= 0
    term2_neg_avg= 0

    for k in range(len(Z)):
        # print('k:{}'.format(k))
        z= Z[k]; w= W[k]
        z*= np.sqrt(2)
        negative_improvementk, x2_opt, term1_neg, term2_neg =  two_step_negative_improvement_only(x1, z, model,
                                    f_best, num_inner_opt_restarts, domain, Xt, Yt, noise, kernel, grid_opt_in, D, with_for)


        negative_improvementk*= 1/np.sqrt(np.pi)*w; term1_neg*= 1/np.sqrt(np.pi)*w; term2_neg*= 1/np.sqrt(np.pi)*w;
        negative_improvement+= negative_improvementk
        term1_neg_avg+= term1_neg
        term2_neg_avg+= term2_neg


        x2_opt_list[k, :]= x2_opt[0, :]

    return negative_improvement, x2_opt_list, term1_neg_avg, term2_neg_avg

def hermite_gaussian_app_two_step_gradient_only(x1, Z, W, kernel, Xt, Yt, noise, model, f_best, D, x2_opt_list):

    x1 = x1.reshape(1, -1)

    negative_gradient = 0
    term1_avg_neg_grad = 0
    term2_avg_neg_grad = 0

    for k in range(len(Z)):

        x2_opt= x2_opt_list[k, :].reshape(1,-1)
        z= Z[k]; w= W[k]
        z *= np.sqrt(2)

        negative_gradientk, term1_neg_grad, term2_neg_grad = two_step_negative_gradient_only(x1, z, x2_opt, kernel, Xt, Yt, noise, model, f_best, D)

        negative_gradientk *= 1/np.sqrt(np.pi)*w; term1_neg_grad *= 1/np.sqrt(np.pi)*w; term2_neg_grad *= 1/np.sqrt(np.pi)*w
        negative_gradient += negative_gradientk; term1_avg_neg_grad += term1_neg_grad; term2_avg_neg_grad += term2_neg_grad

    return negative_gradient, term1_avg_neg_grad, term2_avg_neg_grad

sys.path.append('../functions/')
from sine import sin
import gpflow as gp

def test_temp_hermite_gradient_approximation_1d_sin():

    D= 1
    noise= 10**(-4)

    domain =[[-2,2]]

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt = np.random.uniform(domain[0][0], domain[0][1], size=(3, 1))
    Yt = sin(Xt)

    f_best = float(np.min(Yt, axis=0))

    kernel = gp.kernels.RBF()
    model = gp.models.GPR((Xt, Yt), kernel);
    model.likelihood.variance.assign(noise)

    x1_test = np.random.uniform(lower, upper, (1, D))

    num_inner_opt_restarts= 0
    grid_opt_in= True

    Z, W = np.polynomial.hermite.hermgauss(20)

    '''herm grad'''
    with_for = False
    x_val_neg, x2_opt_list,term1_neg_avg_x, term2_neg_avg_x = hermite_gaussian_app_two_step_improvement_only(x1_test, Z, W, model, f_best, num_inner_opt_restarts,domain, Xt, Yt,
                                                   noise, kernel, grid_opt_in, D, with_for)

    x_val= -x_val_neg; term1_avg_x= -term1_neg_avg_x; term2_avg_x= -term2_neg_avg_x;

    analytical_neg_deriv, term1_avg_neg_deriv, term2_avg_neg_deriv= hermite_gaussian_app_two_step_gradient_only(x1_test, Z, W, kernel, Xt, Yt, noise, model, f_best, D, x2_opt_list)
    analytical_deriv = -analytical_neg_deriv; term1_avg_deriv = -term1_avg_neg_deriv; term2_avg_deriv = -term2_avg_neg_deriv

    '''approximation'''
    x1_pl= x1_test+0.000000001; x1_mn= x1_test- 0.000000001;


    x_pl_val_neg, _, term1_neg_avg_x_pl, term2_neg_avg_x_pl =  hermite_gaussian_app_two_step_improvement_only(x1_pl, Z, W, model, f_best, num_inner_opt_restarts,
                                              domain, Xt, Yt, noise, kernel, grid_opt_in, D, with_for)


    x_pl_val= -x_pl_val_neg; term1_avg_x_pl= -term1_neg_avg_x_pl; term2_avg_x_pl= -term2_neg_avg_x_pl;


    x_mn_val_neg, __, term1_neg_avg_x_mn, term2_neg_avg_x_mn = hermite_gaussian_app_two_step_improvement_only(x1_mn, Z, W,model, f_best, num_inner_opt_restarts,
                                             domain, Xt, Yt,  noise, kernel, grid_opt_in, D, with_for)


    x_mn_val = -x_mn_val_neg; term1_avg_x_mn= -term1_neg_avg_x_mn; term2_avg_x_mn= -term2_neg_avg_x_mn;


    app_deriv_term1 = ((term1_avg_x_pl - term1_avg_x) / (x1_pl - x1_test) + \
                       (term1_avg_x - term1_avg_x_mn) / (x1_test - x1_mn)) / 2

    app_deriv_term2 = ((term2_avg_x_pl - term2_avg_x) / (x1_pl - x1_test) + \
                       (term2_avg_x - term2_avg_x_mn) / (x1_test - x1_mn)) / 2

    app_deriv= ((x_pl_val- x_val )/(x1_pl- x1_test)+\
                                    (x_val- x_mn_val)/(x1_test- x1_mn))/2



    print('\nherm_deriv_term1:{}, app_deriv_term1:{}'.format(term1_avg_deriv, app_deriv_term1))
    print('\nherm_deriv_term2:{}, app_deriv_term2:{}'.format(term2_avg_deriv, app_deriv_term2))
    print('\nherm_deriv:{}, app_deriv:{}'.format(analytical_deriv, app_deriv))


from branin import *

def test_temp_hermite_gradient_approximation_2d_branin():

    D= 2
    noise= 10**(-4)

    domain =[[-5, 10], [0, 15]]

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt =  np.random.uniform(lower, upper, (3,D))

    Yt = branin(Xt)

    f_best = float(np.min(Yt, axis=0))

    kernel = gp.kernels.RBF()
    model = gp.models.GPR((Xt, Yt), kernel);
    model.likelihood.variance.assign(noise)

    x1_test = np.random.uniform(lower, upper, (1, D))

    num_inner_opt_restarts= 0
    grid_opt_in= True

    Z, W = np.polynomial.hermite.hermgauss(20)

    '''herm grad'''

    x_val_neg, x2_opt_list = hermite_gaussian_app_two_step_improvement_only(x1_test, Z, W, model, f_best, num_inner_opt_restarts,domain, Xt, Yt,
                                                   noise, kernel, grid_opt_in, D)
    x_val= -x_val_neg

    analytical_grad= -hermite_gaussian_app_two_step_gradient_only(x1_test, Z, W, kernel, Xt, Yt, noise, model, f_best, D, x2_opt_list)

    '''approximation'''

    x11= x1_test.copy(); x11[0,0]= x11[0,0]+ 0.0001;
    x12= x1_test.copy(); x12[0,0] = x12[0,0]- 0.0001
    x21= x1_test.copy(); x21[0,1] = x21[0,1]+ 0.0001
    x22= x1_test.copy(); x22[0,1] = x22[0,1]- 0.0001


    x11_val_neg, _ =  hermite_gaussian_app_two_step_improvement_only(x11, Z, W, model, f_best, num_inner_opt_restarts,
                                              domain, Xt, Yt, noise, kernel, grid_opt_in, D)


    x11_val= -x11_val_neg

    x12_val_neg, __ = hermite_gaussian_app_two_step_improvement_only(x12, Z, W,model, f_best, num_inner_opt_restarts,
                                             domain, Xt, Yt,  noise, kernel, grid_opt_in, D)


    x12_val = -x12_val_neg

    x21_val_neg, _ =  hermite_gaussian_app_two_step_improvement_only(x21, Z, W, model, f_best, num_inner_opt_restarts,
                                              domain, Xt, Yt, noise, kernel, grid_opt_in, D)


    x21_val= -x21_val_neg

    x22_val_neg, _ =  hermite_gaussian_app_two_step_improvement_only(x22, Z, W, model, f_best, num_inner_opt_restarts,
                                              domain, Xt, Yt, noise, kernel, grid_opt_in, D)


    x22_val= -x22_val_neg

    grad_app_dir1 = ((x11_val - x_val) / (x11[0, 0] - x1_test[0, 0]) + (x_val - x12_val) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2 = ((x21_val - x_val) / (x21[0, 1] - x1_test[0, 1]) + (x_val - x22_val) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app = np.append(grad_app_dir1, grad_app_dir2, axis=0)

    print('herm_grad: \n{}, \ngrad_app:\n{}'.format(analytical_grad, grad_app))
