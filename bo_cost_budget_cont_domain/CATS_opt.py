
import numpy as np
from gp_gradients import *
from scipy.optimize import minimize
from scipy.optimize import Bounds


def CATS_optimize(f_best, Xt, Yt, Yt_latent_cost, model, latent_cost_model, domain, noise, kernel, latent_cost_kernel,
                        num_inner_opt_restarts, num_outer_opt_restarts, D, num_iter_max, grid_opt_in, grid_opt_out):



    def CATS_negative_improvement(x1, Z):

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

        '''cost mean and variance'''
        u0_x1_cost_latent, var0_x1_cost_latent = latent_cost_model.predict_f(x1);
        u0_x1_cost_latent = u0_x1_cost_latent.numpy();
        var0_x1_cost_latent = var0_x1_cost_latent.numpy()
        sigma0_x1_cost_latent = np.sqrt(var0_x1_cost_latent)

        L0_x1=  np.linalg.cholesky(var0_x1)
        y1= u0_x1+ np.matmul(L0_x1,Z_conv)
        f_best1= np.minimum(f_best, float(y1))

        '''update Xt and Yt according to outcome of Z'''
        Xt1= np.append(Xt, x1, axis=0)
        Yt1= np.append(Yt, y1, axis=0)

        '''first two step improvement term'''
        i0_x1= np.maximum(f_best- float(np.min(Yt1, axis=0)), 0)

        x2_opt_value, x2_opt, ei1_x2_opt = EI1_x2_per_cost_optimize(u0_x1, var0_x1, kernel, latent_cost_kernel, x1, Z, Xt, Yt, Yt_latent_cost,
                                 noise, model, latent_cost_model, f_best, domain, num_inner_opt_restarts, grid_opt_in, D, f_best1, num_iter_max)

        #x2_opt_value: EI1_x2_opt_per_cost

        '''second two step improvement term'''
        ei1_x2= ei1_x2_opt
        ei1_x2= float(ei1_x2)

        two_opt= i0_x1+ ei1_x2

        '''total two step cost'''
        u0_x1_cost= np.exp(u0_x1_cost_latent)
        u0_x2_opt_cost_latent, var0_x2_opt_cost_latent = latent_cost_model.predict_f(x2_opt);
        u0_x2_opt_cost_latent = u0_x2_opt_cost_latent.numpy(); var0_x2_cost_latent = var0_x2_opt_cost_latent.numpy()
        u0_x2_opt_cost= np.exp(u0_x2_opt_cost_latent)
        total_cost= u0_x1_cost+ u0_x2_opt_cost


        CATS_x1= two_opt/total_cost
        CATS_x1= CATS_x1.flatten()

        two_opt= np.atleast_2d(two_opt)
        return -CATS_x1, x2_opt, two_opt

    def CATS_negative_gradient(x1, Z, x2_opt, two_opt):

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

        '''gradient of nominator'''
        two_step_grad= term_grad1+ EI1_x2_grad_x1

        '''gradient of total_cost'''

        #gradient of mean cost

        u0_x1_latent_cost, var0_x1_latent_cost= latent_cost_model.predict_f(x1); u0_x1_latent_cost= u0_x1_latent_cost.numpy();
        var0_x1_latent_cost= var0_x1_latent_cost.numpy()
        u0_x1_cost= np.exp(u0_x1_latent_cost)

        du0_x1_lat, dvar0_x1_lat, _= mean_variance_gradients(latent_cost_kernel, Xt, Yt_latent_cost, x1, noise)

        du0_x1_cost= du0_x1_lat* u0_x1_cost

        #ignore gradient of second term

        gradient_total_cost= du0_x1_cost

        '''total cost'''
        u0_x2_opt_latent_cost, var0_x2_opt_latent_cost= latent_cost_model.predict_f(x2_opt); u0_x2_opt_latent_cost= u0_x2_opt_latent_cost.numpy();
        var0_x2_opt_latent_cost= var0_x2_opt_latent_cost.numpy()
        u0_x2_opt_cost= np.exp(u0_x2_opt_latent_cost)

        total_cost= u0_x1_cost+ u0_x2_opt_cost

        '''overall gradient'''
        CATS_grad= (total_cost*two_step_grad- gradient_total_cost*two_opt)/(total_cost**2)
        CATS_grad= CATS_grad.flatten()
        # print('two_step_grad:{}'.format(two_step_grad))
        return -CATS_grad


    monte_carlo_samples= 100
    Z_list = np.random.normal(0.0, 1.0, (1, monte_carlo_samples))


    def monte_carlo_average_two_step_pu_improvement(x1):

        avg_negative_improvement_pu= 0

        global x2_opt_list_mont
        global x2_opt_two_opt_list_mont

        x2_opt_list_mont= np.zeros([Z_list.shape[1], D])
        x2_opt_two_opt_list_mont= np.zeros([Z_list.shape[1], 1])

        for k in range(Z_list.shape[1]):

            z= Z_list[:, k]
            negative_improvement_puk, x2_opt, x2_opt_two_opt = CATS_negative_improvement(x1,z)
            avg_negative_improvement_pu+= negative_improvement_puk/monte_carlo_samples

            x2_opt_list_mont[k, :]= x2_opt[0, :]
            x2_opt_two_opt_list_mont[k, :]= x2_opt_two_opt[0, :]


        return avg_negative_improvement_pu

    def monte_carlo_average_two_step_pu_gradient(x1):

        avg_negative_gradient_pu = 0

        global x2_opt_list_mont
        global x2_opt_two_opt_list_mont

        for k in range(Z_list.shape[1]):

            x2_opt= x2_opt_list_mont[k, :].reshape(1,-1)
            x2_opt_two_opt= x2_opt_two_opt_list_mont[k, :].reshape(1,-1)

            z = Z_list[:, k]
            negative_improvement_puk = CATS_negative_gradient(x1, z, x2_opt, x2_opt_two_opt)
            avg_negative_gradient_pu += negative_improvement_puk / monte_carlo_samples

        return avg_negative_gradient_pu

    Z, W= np.polynomial.hermite.hermgauss(20)

    def hermite_gaussian_app_two_step_pu_improvement(x1):
        print('improvement')
        if type(x1)== np.ndarray:
            x1 = x1.reshape(1, -1)
        elif tf.is_tensor:
            x1= x1.numpy()
            flag_tens= True

        negative_improvement_pu= 0

        global x2_opt_list
        global x2_opt_two_opt_list_herm

        x2_opt_list= np.zeros([len(Z), D])
        x2_opt_two_opt_list_herm = np.zeros([len(Z), 1])

        for k in range(len(Z)):
            # print('k:{}'.format(k))
            z= Z[k]; w= W[k]
            z*= np.sqrt(2)
            negative_improvement_puk, x2_opt, x2_opt_two_opt = CATS_negative_improvement(x1, z)
            negative_improvement_puk*= 1/np.sqrt(np.pi)*w
            negative_improvement_pu+= negative_improvement_puk

            x2_opt_list[k, :]= x2_opt[0, :]
            x2_opt_two_opt_list_herm[k, :] = x2_opt_two_opt[0, :]
        # print('negative_improvement_per_cost:{}'.format(negative_improvement_pu))

        return negative_improvement_pu

    def hermite_gaussian_app_two_step_pu_gradient(x1):
        print('gradient')
        x1 = x1.reshape(1, -1)

        negative_gradient_pu = 0

        global x2_opt_list
        global x2_opt_two_opt_list_herm

        for k in range(len(Z)):

            x2_opt= x2_opt_list[k, :].reshape(1,-1)
            x2_opt_two_opt = x2_opt_two_opt_list_herm[k, :].reshape(1, -1)

            z= Z[k]; w= W[k]
            z *= np.sqrt(2)

            negative_improvement_puk = CATS_negative_gradient(x1,z, x2_opt, x2_opt_two_opt)
            negative_improvement_puk *= 1/np.sqrt(np.pi)*w
            negative_gradient_pu += negative_improvement_puk

        return negative_gradient_pu

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
            result= (minimize(hermite_gaussian_app_two_step_pu_improvement, x10, bounds=b,  method='L-BFGS-B',
                              jac=hermite_gaussian_app_two_step_pu_gradient, options= {'maxiter': num_iter_max}))
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
                x1_cand_value= -hermite_gaussian_app_two_step_pu_improvement(x1_cand);
                x1_cand_value= x1_cand_value.reshape(1,-1)
                x1_value_list_grid[j, :]= x1_cand_value[0,:]

            index_max_grid = int(np.argmax(x1_value_list_grid, axis=0))
            x_opt_value_grid =x1_value_list_grid[index_max_grid, :].reshape(1, -1)
            x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)

        if D == 2 and grid_opt_out== True:
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
                x1_cand_value = -hermite_gaussian_app_two_step_pu_improvement(x1_cand);
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

sys.path.append('../functions')
from sine import *
sys.path.append('../cost_functions')
from cos_1d import *
from exp_cos_1d import *

def temp_test_cats_opt_1d():

    _, __, domain = sin_opt()
    D = 1; noise = 10 ** (-4)
    _, __, domain_cost= exp_cos_1d_opt()
    noise_cost = 10 ** (-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= sin(Xt)
    Yt_cost= exp_cos_1d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 0; num_outer_opt_restarts=0; D=1;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    Z, W = np.polynomial.hermite.hermgauss(20)

    x1_test = np.random.uniform(lower, upper, (1, D))


    x1_test_val= -hermite_gaussian_app_two_step_pu_improvement(x1_test)

    x2_opt_list_test= x2_opt_list.copy()

    '''analytical gradient'''
    x1_test_grad= -hermite_gaussian_app_two_step_pu_gradient(x1_test)



    '''approximate gradient'''
    x1_pl = x1_test + 0.00001; x1_mn = x1_test - 0.00001;

    x1_pl_val= -hermite_gaussian_app_two_step_pu_improvement(x1_pl)
    x2_opt_list_pl = x2_opt_list.copy()

    x1_mn_val= -hermite_gaussian_app_two_step_pu_improvement(x1_mn)
    x2_opt_list_mn = x2_opt_list.copy()

    app_deriv= ((x1_pl_val- x1_test_val )/(x1_pl- x1_test)+\
                                    (x1_test_val- x1_mn_val)/(x1_test- x1_mn))/2

    print('x2_opt_list_test, x2_opt_list_pl, x2_opt_list_mn\n',
            np.append(np.append(x2_opt_list_test, x2_opt_list_pl, axis=1), x2_opt_list_mn, axis=1))
    print('\nanalytical_deriv:{}, app_deriv:{}'.format(x1_test_grad, app_deriv))


sys.path.append('../functions')
sys.path.append('../cost_functions')
from exp_cos_2d import *
from branin import *

def temp_test_cats_opt_2d():

    _, __, domain = branin_opt()
    D = 2; noise = 10 ** (-4)
    _, __, domain_cost= exp_cos_2d_opt()
    noise_cost = 10 ** (-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= branin(Xt)
    Yt_cost= exp_cos_2d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 0; num_outer_opt_restarts=0;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    Z, W = np.polynomial.hermite.hermgauss(20)

    x1_test = np.random.uniform(lower, upper, (1, D))


    x1_test_val= -hermite_gaussian_app_two_step_pu_improvement(x1_test)

    x2_opt_list_test= x2_opt_list.copy()

    '''analytical gradient'''
    x1_test_grad= -hermite_gaussian_app_two_step_pu_gradient(x1_test)


    '''approximate gradient'''

    x11= x1_test.copy(); x11[0,0]= x11[0,0]+ 0.0001;
    x12= x1_test.copy(); x12[0,0] = x12[0,0]- 0.0001
    x21= x1_test.copy(); x21[0,1] = x21[0,1]+ 0.0001
    x22= x1_test.copy(); x22[0,1] = x22[0,1]- 0.0001


    x11_val=   -hermite_gaussian_app_two_step_pu_improvement(x11)
    x12_val=   -hermite_gaussian_app_two_step_pu_improvement(x12)
    x21_val=   -hermite_gaussian_app_two_step_pu_improvement(x21)
    x22_val=   -hermite_gaussian_app_two_step_pu_improvement(x22)


    grad_app_dir1 = ((x11_val -x1_test_val) / (x11[0, 0] - x1_test[0, 0]) + (x1_test_val - x12_val) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2 = ((x21_val - x1_test_val) / (x21[0, 1] - x1_test[0, 1]) + (x1_test_val - x22_val) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app = np.append(grad_app_dir1, grad_app_dir2, axis=0)

    print('analytical_grad: \n{}, \napp_grad:\n{}'.format(x1_test_grad, grad_app))

import time
from tensorflow.test import compute_gradient

def temp_speed_test():


    _, __, domain = branin_opt()
    D = 2; noise = 10 ** (-4)
    _, __, domain_cost= exp_cos_2d_opt()
    noise_cost = 10 ** (-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= branin(Xt)
    Yt_cost= exp_cos_2d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 0; num_outer_opt_restarts=0;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    Z, W = np.polynomial.hermite.hermgauss(20)

    x1_test = np.random.uniform(lower, upper, (1, D))

    '''calculate function evaluation time'''
    t1_fun= time.clock()
    x1_test_val= -hermite_gaussian_app_two_step_pu_improvement(x1_test)
    t2_fun= time.clock()

    print('function evaluation time:{}'.format(t2_fun- t1_fun))
    x2_opt_list_test= x2_opt_list.copy()

    '''calculate gradient evaluation time'''
    t1_grad= time.clock()
    x1_test_grad= -hermite_gaussian_app_two_step_pu_gradient(x1_test)
    t2_grad = time.clock()

    print('gradient evaluation time_code:{}'.format(t2_grad - t1_grad))

    x1_test_tens= tf.Variable(x1_test)
    t1_grad_tens = time.clock()
    tf.test.compute_gradient(hermite_gaussian_app_two_step_pu_improvement, [x1_test_tens])
    t2_grad_tens = time.clock()

    print('gradient evaluation time tensorflow_jacobian:{}'.format(t2_grad_tens - t1_grad_tens))

    '''use gradient tape'''
    with tf.GradientTape(persistent= True) as g:
        g.watch(x1_test_tens)
        t1_fun= time.clock()
        x1_test_val= -hermite_gaussian_app_two_step_pu_improvement(x1_test_tens)
        x1_temp= x1_test_tens.numpy()
        y_test= x1_temp**2
        y_test= tf.constant(y_test)

        x1_test_val_tens= tf.constant(x1_test_val)
        t2_fun= time.clock()

    t1_tens_grad= time.clock()
    dy_dx = g.gradient(x1_test_val_tens, x1_test_tens)
    t2_tens_grad= time.clock()

    dy_dx_test = g.gradient(y_test, x1_test_tens)

    print('gradient evaluation time tensorflow_gradient_tape:{}'.format(t2_tens_grad - t1_tens_grad))

    print('tensorflow_gradient:{}, code_gradient:{}'.format(dy_dx, x1_test_grad))


    '''time for an optimization step'''
    lower = [];
    upper = []
    for i in range(len(domain)):
        lower.append(domain[i][0])
        upper.append(domain[i][1])
    b = Bounds(lb=lower, ub=upper)
    t1_min= time.clock()
    for i in range(10):
        result = (minimize(hermite_gaussian_app_two_step_pu_improvement, x1_test, bounds=b, method='L-BFGS-B',
                           jac=hermite_gaussian_app_two_step_pu_gradient, options={'maxiter':100}))
    t2_min= time.clock()

    t1_cats_opt= time.clock()
    CATS_optimize(f_best, Xt, Yt, Yt_latent_cost, model, latent_cost_model, domain, noise, kernel, latent_cost_kernel,
                  0, 10, D, 100, True, False)
    t2_cats_opt= time.clock()

    print('scipy minimize time:{}, cats_optimize:{}'.format((t2_min- t1_min), (t2_cats_opt-t1_cats_opt)))


def temp_test_lbfgs():

    _, __, domain = branin_opt()
    D = 2; noise = 10 ** (-4)
    _, __, domain_cost= exp_cos_2d_opt()
    noise_cost = 10 ** (-4)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= branin(Xt)
    Yt_cost= exp_cos_2d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 0; num_outer_opt_restarts=0;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    Z, W = np.polynomial.hermite.hermgauss(20)

    x1_test = np.random.uniform(lower, upper, (1, D))