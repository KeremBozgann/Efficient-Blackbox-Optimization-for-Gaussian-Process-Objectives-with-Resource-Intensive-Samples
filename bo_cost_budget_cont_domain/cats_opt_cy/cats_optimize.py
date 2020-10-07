
import numpy as np
# from gp_gradients import *
from scipy.optimize import minimize
from scipy.optimize import Bounds
from ei_per_cost_optimize import *
from scipy.stats import norm
import tensorflow as tf

def funfi(A):

    result= np.tril(A, k=-1)+ 1/2*np.diagflat(np.diag(A))
    return result

def get_posterior_covariance_and_derivative_1q(x, x1, Xt, kernel, noise):

    #K_x1_x= K_x_x1

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    k_x1_Xt= (kernel.K(x1,Xt)).numpy()
    k_x_Xt_T= np.transpose((kernel.K(x, Xt)).numpy())
    alpha= np.matmul(K_inv, k_x_Xt_T)
    k_x1_x= (kernel.K(x1,x)).numpy()
    K_x1_x= (k_x1_x- np.matmul(k_x1_Xt, alpha))


    '''gradient for 1d'''
    ls = (kernel.lengthscales._value()).numpy()

    if type(ls)== np.ndarray:

        lamb= np.diagflat(ls)
        lamb_inv= np.diagflat(1/ls)
    else:

        lamb= np.eye(Xt.shape[1])*ls
        lamb_inv= np.eye(Xt.shape[1])*1/ls

    k_x1_x= (kernel.K(x1,x)).numpy()
    diff_vec= (x-x1).T
    term1= np.matmul(lamb_inv, (diff_vec* k_x1_x))
    term2 = -np.matmul(lamb_inv, np.matmul(np.transpose(x1-Xt)*k_x1_Xt, alpha))
    dK_x1_x_dx1= term1-term2

    return K_x1_x, dK_x1_x_dx1

def get_dsigma0_dx1_q1(x2_opt, x1, Xt, kernel, noise, K0_x1_x1,  dK0_x1_dx1, D, Yt, L0_x1, L0_x1_inv, L0_x1_inv_T):

    #K0_x1: 1*1, dK0_x1_dx1: D*1
    K0_x2_x1, dK0_x2_x1_dx1 = get_posterior_covariance_and_derivative_1q(x2_opt, x1, Xt, kernel, noise)
    # _, dK0_x1_dx1, K0_x1_x1= mean_variance_gradients(kernel, Xt, Yt, x1, noise)

    # L0_x1= np.linalg.cholesky(K0_x1_x1)
    # L0_x1_inv= np.linalg.inv(L0_x1); L0_x1_inv_T= np.transpose(L0_x1_inv)

    dL0_x1_dx1 = np.empty([D, 1])
    for i in range(D):
        dK0_x1_dx1i = dK0_x1_dx1[i, :].reshape(1, 1)
        Ai = np.matmul(L0_x1_inv, np.matmul(dK0_x1_dx1i, L0_x1_inv_T))
        temp = funfi(Ai)
        dL0_x1_dx1i = np.matmul(L0_x1, temp)
        dL0_x1_dx1[i, 0] = dL0_x1_dx1i

    '''only for q=1'''
    dL0_x1_inv_dx1= np.empty([D,1])
    for i in range(D):
        dL0_x1_dx1i= dL0_x1_dx1[i,:].reshape(1,1)
        dL0_x1_inv_dx1i= -np.matmul(L0_x1_inv, np.matmul(dL0_x1_dx1i, L0_x1_inv))
        dL0_x1_inv_dx1[i,0]= dL0_x1_inv_dx1i

    '''only for q=1'''
    dsigma0_dx1= np.empty([D,1])
    for i in range(D):
        dK0_x2_x1_dx1i= dK0_x2_x1_dx1[i,:].reshape(1,1);
        dL0_x1_inv_dx1i= dL0_x1_inv_dx1[i,:].reshape(1,1)
        dsigma0_dx1i= np.matmul(np.transpose(L0_x1_inv),dK0_x2_x1_dx1i)+ np.matmul(K0_x2_x1, dL0_x1_inv_dx1i)
        dsigma0_dx1[i,0]= dsigma0_dx1i

    sigma0= np.matmul(K0_x2_x1, L0_x1_inv)

    return dsigma0_dx1, sigma0, dL0_x1_dx1

def get_lookahead_mean_and_variance_gradient_wrt_x1(dsigma0_dx1, sigma0, Z, D):

    du1_x2_dx1= np.matmul(dsigma0_dx1, Z)

    dvar1_x2_dx1= np.empty([D,1])

    '''holds also for q>1'''
    for i in range(D):
        dsigma0_dx1i= dsigma0_dx1[i, :].reshape(-1,1)
        dvar1_x2_dx1i= -2* np.matmul(sigma0, np.transpose(dsigma0_dx1i))
        dvar1_x2_dx1[i, 0]= dvar1_x2_dx1i

    return du1_x2_dx1, dvar1_x2_dx1

def get_df_best1_dx1(f_best, Z, du0_x1_dx1, dL0_x1_dx1, y1):

    if (f_best- y1) <0:

        best1_grad= 0

    else:
        best1_grad= du0_x1_dx1+ np.matmul(dL0_x1_dx1, Z)

    return best1_grad

def get_one_step_mean_and_variance(u0_x2, var0_x2, Z, sigma0):

    u1_x2= u0_x2 + np.matmul(sigma0,Z)

    var1_x2= var0_x2- np.matmul(sigma0, sigma0.T)


    return u1_x2, var1_x2


def dnormal_dfbest(f_best, u_x, sigma_x):

    return -(f_best - u_x) / (sigma_x** 2) * norm.pdf((f_best - u_x) / sigma_x)

def cats_dEI1_x2_dx1(f_best, f_best1,  kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x2, var0_x2, L0_x1, L0_x1_inv, L0_x1_inv_T, y1):

    du0_x1_dx1, dK0_x1_dx1, K0_x1 =  get_mean_variance_gradients_q1(kernel, Xt, Yt, x1, noise)
    dsigma0_dx1, sigma0, dL0_x1_dx1= get_dsigma0_dx1_q1(x2_opt, x1, Xt, kernel, noise, K0_x1, dK0_x1_dx1, D, Yt,
                                                           L0_x1, L0_x1_inv, L0_x1_inv_T)

    du1_x2_opt_dx1, dK1_x2_opt_dx1= get_lookahead_mean_and_variance_gradient_wrt_x1(dsigma0_dx1, sigma0, Z, D)

    best1_grad_x1= get_df_best1_dx1(f_best, Z, du0_x1_dx1, dL0_x1_dx1, y1)

    u1_x2_opt, var1_x2_opt= get_one_step_mean_and_variance(u0_x2, var0_x2, Z, sigma0)
    sigma1_x2_opt= np.sqrt(var1_x2_opt)
    dsigma1_x2_opt_dx1 = (1 / (2 * sigma1_x2_opt)) * dK1_x2_opt_dx1

    '''cumulative gradient'''
    dcumulative_dx= dcumulative_du(f_best1, sigma1_x2_opt,u1_x2_opt)* du1_x2_opt_dx1 + \
                            dcumulative_dsigma(f_best1,sigma1_x2_opt,u1_x2_opt)*dsigma1_x2_opt_dx1+\
                                dcumulative_dfbest(f_best1,sigma1_x2_opt,u1_x2_opt)*best1_grad_x1


    term1= (f_best1- u1_x2_opt)*dcumulative_dx

    term2= (best1_grad_x1-du1_x2_opt_dx1)* norm.cdf(((f_best1- u1_x2_opt)/sigma1_x2_opt))

    term3= dsigma1_x2_opt_dx1* norm.pdf((f_best1- u1_x2_opt)/sigma1_x2_opt)

    '''normal distribution gradient'''

    dnormal_dx= dnormal_du(f_best1, u1_x2_opt, sigma1_x2_opt)*du1_x2_opt_dx1+ \
                                dnormal_dsigma(f_best1, u1_x2_opt, sigma1_x2_opt)*dsigma1_x2_opt_dx1+\
                        dnormal_dfbest(f_best1, u1_x2_opt, sigma1_x2_opt)*best1_grad_x1

    term4= sigma1_x2_opt* dnormal_dx

    return term1+ term2+ term3+ term4


class cats_optimize():


    def __init__(self):

        pass

    def update_x1_dependent(self, model, latent_cost_model, x1):

        self.u0_x1, self.var0_x1, self.sigma0_x1= get_mean_var_std(x1, model)

        self.u0_x1_cost = get_mean_of_cost(x1, latent_cost_model)

    def update_z_dependent(self, z, f_best, Xt, Yt, x1):

        self.L0_x1=  np.linalg.cholesky(self.var0_x1)
        self.y1= self.u0_x1+ np.matmul(self.L0_x1,z)
        self.f_best1= np.minimum(f_best, float(self.y1))

        '''update Xt and Yt according to outcome of Z'''
        self.Xt1= np.append(Xt, x1, axis=0)
        self.Yt1= np.append(Yt, self.y1, axis=0)

        '''first two step improvement term'''
        self.i0_x1 = np.maximum(f_best - float(np.min(self.Yt1, axis=0)), 0)

    def cats_negative_improvement(self, x1, kernel, latent_cost_kernel, z, num_inner_opt_restarts, grid_opt_in, D, Xt, Yt,
                                  noise, model, latent_cost_model, domain, Yt_latent_cost, num_iter_max, f_best, noise_cost):

        x1 = x1.reshape(1,-1)
        # print('x1 shape:{}'.format(x1.shape))

        '''first ts term'''
        opt = Ei_per_cost_optimize(self.u0_x1, self.var0_x1, x1, z, Xt, Yt)
        self.x2_opt_value, self.x2_opt, self.ei1_x2_opt= opt.maximize_ei_pu(kernel, latent_cost_kernel, x1, z, Xt, noise, model, latent_cost_model, domain,
                             num_inner_opt_restarts, grid_opt_in, D, self.f_best1, Yt_latent_cost, num_iter_max, noise_cost)
        #x2_opt_value: EI1_x2_opt_per_cost

        '''second two step improvement term'''
        self.ei1_x2= self.ei1_x2_opt
        self.ei1_x2= float(self.ei1_x2)

        self.two_opt= self.i0_x1+ self.ei1_x2

        '''total two step cost'''

        self.u0_x2_opt_cost= get_mean_of_cost(self.x2_opt, latent_cost_model)
        self.total_cost= self.u0_x1_cost+ self.u0_x2_opt_cost

        CATS_x1= self.two_opt/self.total_cost
        CATS_x1= CATS_x1.flatten()

        self.two_opt= np.atleast_2d(self.two_opt)

        return -CATS_x1, self.x2_opt, self.two_opt

    def cats_negative_gradient(self, x1, kernel, latent_cost_kernel, z, num_inner_opt_restarts, grid_opt_in, D, Xt, Yt,
                                  noise, model, latent_cost_model, domain, Yt_latent_cost, num_iter_max, f_best, noise_cost):

        x1 = x1.reshape(1,-1)

        '''find cholesky gradient (L0_x1) and du0_x1_dx1'''
        self.du0_x1_dx1, self.dvar0_x1_dx1, self.K0_x1 = get_mean_variance_gradients_q1(kernel, Xt, Yt, x1, noise)

        self.L0_x1_inv = np.linalg.inv(self.L0_x1);
        self.L0_x1_inv_T = np.transpose(self.L0_x1_inv)

        if f_best- self.y1<0:
            term_grad1= 0

        else:

            # _, dK0_x1_dx1, K0_x1_x1 = mean_variance_gradients(kernel, Xt, Yt, x1, noise)

            self.dL0_x1_dx1= np.empty([D,1])
            for i in range(D):
                dvar0_x1_dx1i= self.dvar0_x1_dx1[i,:].reshape(1,1)
                Ai= np.matmul(self.L0_x1_inv, np.matmul(dvar0_x1_dx1i, self.L0_x1_inv_T))
                temp= funfi(Ai)
                dL0_x1_dx1i = np.matmul(self.L0_x1, temp)
                self.dL0_x1_dx1[i,0]= dL0_x1_dx1i

            # term_grad1= -df_best1_dx1(f_best, x1, u0_x1, L0_x1, Z, du0_x1_dx1, dL0_x1_dx1)
            term_grad1= -self.du0_x1_dx1- np.matmul(self.dL0_x1_dx1, z)

        # global x2_opt

        '''gradient of lookahead term'''
        u0_x2_opt, var0_x2_opt, sigma0_x2_opt= get_mean_var_std(self.x2_opt, model)

        EI1_x2_grad_x1= cats_dEI1_x2_dx1(f_best, self.f_best1, kernel, Xt, Yt, noise, self.x2_opt, x1, D, z, u0_x2_opt,
                    var0_x2_opt, self.L0_x1, self.L0_x1_inv, self.L0_x1_inv_T, self.y1)

        # print('term_grad1:{}, EI1_x2_grad_x1:{}'.format(term_grad1, EI1_x2_grad_x1))

        '''gradient of nominator'''
        two_step_grad= term_grad1+ EI1_x2_grad_x1

        '''gradient of total_cost'''

        #gradient of mean cost

        du0_x1_lat, dvar0_x1_lat, _= get_mean_variance_gradients_q1(latent_cost_kernel, Xt, Yt_latent_cost, x1, noise_cost)

        du0_x1_cost= du0_x1_lat* self.u0_x1_cost

        #ignore gradient of second term

        gradient_total_cost= du0_x1_cost

        '''overall gradient'''
        CATS_grad= (self.total_cost*two_step_grad- gradient_total_cost*self.two_opt)/(self.total_cost**2)
        CATS_grad= CATS_grad.flatten()
        # print('two_step_grad:{}'.format(two_step_grad))
        return -CATS_grad

    def set_monte_carlo_parameters(self, monte_carlo_samples):

        self.monte_carlo_samples= 100
        self.Z_list = np.random.normal(0.0, 1.0, (1, monte_carlo_samples))


    def monte_carlo_average_two_step_pu_improvement(self, x1, model, latent_cost_model, f_best, Xt, Yt, kernel, latent_cost_kernel,
                                                    num_inner_opt_restarts, grid_opt_in, noise, domain, Yt_latent_cost,
                                                    num_iter_max, D):
        x1 = x1.reshape(1, -1)
        avg_negative_improvement_pu= 0


        self.x2_opt_list_mont= np.zeros([self.Z_list.shape[1], D])
        self.x2_opt_two_opt_list_mont= np.zeros([self.Z_list.shape[1], 1])

        self.update_x1_dependent(model, latent_cost_model, x1)
        for k in range(self.Z_list.shape[1]):

            z= self.Z_list[:, k]
            z= np.atleast_2d(z)

            self.update_z_dependent(z, f_best, Xt, Yt, x1)
            negative_improvement_puk, x2_opt, x2_opt_two_opt = self.cats_negative_improvement(x1, kernel, latent_cost_kernel,
                                    z, num_inner_opt_restarts, grid_opt_in, D, Xt, Yt, noise, model, latent_cost_model,
                                                                    domain, Yt_latent_cost, num_iter_max, f_best)

            avg_negative_improvement_pu+= negative_improvement_puk/self.monte_carlo_samples

            self.x2_opt_list_mont[k, :]= x2_opt[0, :]
            self.x2_opt_two_opt_list_mont[k, :]= x2_opt_two_opt[0, :]


        return avg_negative_improvement_pu

    def monte_carlo_average_two_step_pu_gradient(self, x1, model, latent_cost_model, f_best, Xt, Yt, kernel, latent_cost_kernel,
                                                    num_inner_opt_restarts, grid_opt_in, noise, domain, Yt_latent_cost,
                                                    num_iter_max, D):

        x1 = x1.reshape(1, -1)

        avg_negative_gradient_pu = 0
        self.update_x1_dependent(model, latent_cost_model, x1)

        for k in range(self.Z_list.shape[1]):
            self.x2_opt= self.x2_opt_list_mont[k, :].reshape(1,-1)
            self.x2_opt_two_opt= self.x2_opt_two_opt_list_mont[k, :].reshape(1,-1)

            z = self.Z_list[:, k]
            z= np.atleast_2d(z)

            self.update_z_dependent(z, f_best, Xt, Yt, x1)
            negative_improvement_puk = self.cats_negative_gradient(self, x1, kernel, latent_cost_kernel, z,
                                   num_inner_opt_restarts, grid_opt_in, D, Xt, Yt, noise, model, latent_cost_model,
                                          domain, Yt_latent_cost, num_iter_max, f_best)

            avg_negative_gradient_pu += negative_improvement_puk / self.monte_carlo_samples

        return avg_negative_gradient_pu

    def set_hermite_gauss_parameters(self, n):
        self.Z, self.W= np.polynomial.hermite.hermgauss(n)

    def hermite_gaussian_app_two_step_pu_improvement(self, x1, D, model, latent_cost_model, f_best, Xt, Yt, kernel, latent_cost_kernel,
                                                     num_inner_opt_restarts, grid_opt_in, noise, domain, Yt_latent_cost, num_iter_max, noise_cost):

        print('function')
        if type(x1)== np.ndarray:
            x1 = x1.reshape(1, -1)
        elif tf.is_tensor:
            x1= x1.numpy()
            flag_tens= True

        negative_improvement_pu= 0

        self.x2_opt_list= np.zeros([len(self.Z), D])
        self.x2_opt_two_opt_list_herm = np.zeros([len(self.Z), 1])

        self.update_x1_dependent(model, latent_cost_model, x1)

        for k in range(len(self.Z)):
            # print('k:{}'.format(k))
            z= self.Z[k]; w= self.W[k]
            z*= np.sqrt(2)
            z= np.atleast_2d(z)
            self.update_z_dependent(z, f_best, Xt, Yt, x1)
            negative_improvement_puk, x2_opt, x2_opt_two_opt = self.cats_negative_improvement(x1, kernel, latent_cost_kernel, z, num_inner_opt_restarts, grid_opt_in, D, Xt, Yt,
                                  noise, model, latent_cost_model, domain, Yt_latent_cost, num_iter_max, f_best, noise_cost)

            negative_improvement_puk*= 1/np.sqrt(np.pi)*w
            negative_improvement_pu+= negative_improvement_puk

            self.x2_opt_list[k, :]= x2_opt[0, :]
            self.x2_opt_two_opt_list_herm[k, :] = x2_opt_two_opt[0, :]
        # print('negative_improvement_per_cost:{}'.format(negative_improvement_pu))

        return negative_improvement_pu

    def hermite_gaussian_app_two_step_pu_gradient(self, x1, D, model, latent_cost_model, f_best, Xt, Yt, kernel, latent_cost_kernel,
                                                     num_inner_opt_restarts, grid_opt_in, noise, domain, Yt_latent_cost, num_iter_max, noise_cost):
        print('gradient')

        if type(x1)== np.ndarray:
            x1 = x1.reshape(1, -1)
        elif tf.is_tensor:
            x1= x1.numpy()
            flag_tens= True

        negative_gradient_pu = 0
        self.update_x1_dependent(model, latent_cost_model, x1)

        for k in range(len(self.Z)):

            self.x2_opt= self.x2_opt_list[k, :].reshape(1,-1)
            self.x2_opt_two_opt = self.x2_opt_two_opt_list_herm[k, :].reshape(1, -1)

            z= self.Z[k]; w= self.W[k]
            z *= np.sqrt(2)
            z = np.atleast_2d(z)
            self.update_z_dependent(z, f_best, Xt, Yt, x1)
            negative_improvement_puk = self.cats_negative_gradient(x1, kernel, latent_cost_kernel, z,
                                num_inner_opt_restarts, grid_opt_in, D, Xt, Yt, noise, model, latent_cost_model,
                                                               domain, Yt_latent_cost, num_iter_max, f_best, noise_cost)

            negative_improvement_puk *= 1/np.sqrt(np.pi)*w
            negative_gradient_pu += negative_improvement_puk

        return negative_gradient_pu

    def optimize_cats(self, num_outer_opt_restarts, domain, num_iter_max, grid_opt_out, D, model, latent_cost_model, f_best,
                  Xt, Yt, kernel, latent_cost_kernel, num_inner_opt_restarts, grid_opt_in, noise, Yt_latent_cost, n, noise_cost):


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

            self.set_hermite_gauss_parameters(n)

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

                fun_args= (D, model, latent_cost_model, f_best, Xt, Yt, kernel, latent_cost_kernel,
                                                     num_inner_opt_restarts, grid_opt_in, noise, domain, Yt_latent_cost, num_iter_max, noise_cost)

                result= (minimize(self.hermite_gaussian_app_two_step_pu_improvement, x10, bounds=b,  method='L-BFGS-B',
                                  args= fun_args, jac= self.hermite_gaussian_app_two_step_pu_gradient, options= {'maxiter': num_iter_max}))
                x1_cand_sci= result['x']; cand_value_x1_sci= -result['fun']; x1_cand_grad_sci= -result['jac']
                cand_value_x1_sci= cand_value_x1_sci.reshape(1,-1); x1_cand_grad_sci= x1_cand_grad_sci.reshape(1,-1)
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

                    x1_cand_value= -self.hermite_gaussian_app_two_step_pu_improvement(x1_cand, D, model, latent_cost_model,
                                              f_best, Xt, Yt, kernel, latent_cost_kernel, num_inner_opt_restarts, grid_opt_in,
                                                    noise, domain, Yt_latent_cost, num_iter_max);
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
                    x1_cand_value = -self.hermite_gaussian_app_two_step_pu_improvement(x1_cand, D, model, latent_cost_model,
                                              f_best, Xt, Yt, kernel, latent_cost_kernel, num_inner_opt_restarts, grid_opt_in,
                                                    noise, domain, Yt_latent_cost, num_iter_max)
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
sys.path.append('../')
from CATS_opt import *

def temp_test_speed_1d():

    _, __, domain = sin_opt()
    D = 1; noise = 10 ** (-3)
    _, __, domain_cost= exp_cos_1d_opt()
    noise_cost = 10 ** (-3)

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

    num_inner_opt_restarts= 0; num_outer_opt_restarts=10; D=1;
    grid_opt_in= True; grid_opt_out= False
    num_iter_max = 100;
    n=20
    t1_optimized_py= time.clock()
    opt= cats_optimize()
    opt.optimize_cats(num_outer_opt_restarts, domain, num_iter_max, grid_opt_out, D, model, latent_cost_model, f_best,
                  Xt, Yt, kernel, latent_cost_kernel, num_inner_opt_restarts, grid_opt_in, noise, Yt_latent_cost, n, noise_cost)
    t2_optimized_py= time.clock()

    t1_unoptimized_py= time.clock()
    CATS_optimize(f_best, Xt, Yt, Yt_latent_cost, model, latent_cost_model, domain, noise, kernel, latent_cost_kernel,
                        num_inner_opt_restarts, num_outer_opt_restarts, D, num_iter_max, grid_opt_in, grid_opt_out)
    t2unoptimized_py= time.clock()

    print('unoptimized py code time:{}, optimized py code time:{}'.format((t2unoptimized_py- t1_unoptimized_py),
                                                                          (t2_optimized_py- t1_optimized_py)))

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