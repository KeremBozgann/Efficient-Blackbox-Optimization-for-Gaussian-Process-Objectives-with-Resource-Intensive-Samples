
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import gpflow as gp
# from cats_optimize import get_dsigma0_dx1_q1

def dcumulative_du(f_best, sigma_x, u_x):

    return -1/sigma_x*norm.pdf((f_best- u_x)/sigma_x)


def dcumulative_dsigma(f_best, sigma_x, u_x):

    return -(f_best-u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)


def dnormal_du(f_best, u_x, sigma_x):

    return (f_best- u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)


def dnormal_dsigma(f_best, u_x, sigma_x):

    return ((f_best-u_x)**2)/(sigma_x**3)*norm.pdf((f_best- u_x)/sigma_x)


def ei_pu(sigma_x, u_x, f_best, u_cost):
    gama_x = (f_best - u_x) / sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x = sigma_x * (gama_x * fi_x + norm.pdf(gama_x))

    EI_pu = EI_x / u_cost

    return EI_pu


def ei(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))

    return EI_x


def get_mean_variance_gradients_q1(kernel, Xt, Yt, x, noise):

    '''implement du_dx and gradient of K(x)= k(x,x)- k(x,Xt)(K(Xt,Xt)+I*noise)_inv*k(x,Xt)_T'''
    # K_inv = np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)

    k_x_x= kernel.K(x,x); k_x_x= k_x_x.numpy()
    k_x_Xt= kernel.K(x,Xt); k_x_Xt= k_x_Xt.numpy()
    k_x_Xt_T= np.transpose(k_x_Xt)
    k_Xt_Xt= (kernel.K(Xt,Xt)).numpy(); noise_mat= np.eye(Xt.shape[0])*noise

    K_inv= np.linalg.inv(k_Xt_Xt+noise_mat)

    K_x= k_x_x- np.matmul(k_x_Xt, np.matmul(K_inv, k_x_Xt_T))

    alpha= np.matmul(K_inv,Yt)

    ls = (kernel.lengthscales._value()).numpy()

    if type(ls)== np.ndarray:

        lamb= np.diagflat(ls)
        lamb_inv= np.diagflat(1/ls)
    else:

        lamb= np.eye(Xt.shape[1])*ls
        lamb_inv= np.eye(Xt.shape[1])*1/ls

    X_vec= x- Xt

    # du_dx= np.matmul(np.matmul(-lamb_inv, X_vec.T), kernel.K(Xt, x)* alpha)
    du_dx= np.matmul(np.matmul(-lamb_inv, X_vec.T), k_x_Xt_T* alpha)

    '''variance'''
    # gradient_vec= np.matmul(2*lamb_inv, ((x.T-Xt.T)*(kernel.K(x, Xt).numpy())));
    temp1= np.matmul(2*lamb_inv, ((x.T-Xt.T)*(k_x_Xt)));
    temp2= np.matmul(temp1, K_inv);
    # dvar_dx= np.matmul(temp1, kernel.K(Xt, x))
    dvar_dx= np.matmul(temp2, k_x_Xt_T)


    return du_dx, dvar_dx, K_x


def get_posterior_covariance(x, x1, Xt, kernel, noise):

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    temp= np.matmul((kernel.K(x,Xt)).numpy(), K_inv)
    result= (kernel.K(x,x1)).numpy()- np.matmul(temp, ((kernel.K(x1, Xt)).numpy()).T)

    return result

def get_one_step_mean_and_variance(x2, u0_x2, var0_x2, kernel, x1, Z, Xt, noise, var0_x1):


    k0_x2_x1= get_posterior_covariance(x2, x1, Xt, kernel,  noise)
    # k0_x1_x1= get_posterior_covariance(x1, x1, Xt, kernel, noise)
    L0_x1= np.linalg.cholesky(var0_x1)
    sigma0= np.matmul(k0_x2_x1, np.linalg.inv(L0_x1))

    u1_x2= u0_x2 + np.matmul(sigma0,Z)

    var1_x2= var0_x2- np.matmul(sigma0, sigma0.T)


    return u1_x2, var1_x2


def get_mean_var_std(x2, model):

    u0_x2, var0_x2 = model.predict_f(x2);
    u0_x2 = u0_x2.numpy();
    var0_x2 = var0_x2.numpy()
    sigma0_x2= np.sqrt(var0_x2)

    return u0_x2, var0_x2, sigma0_x2

def get_mean_and_var_cost(x2, latent_cost_model):

    u0_x2_latent_cost, var0_x2_latent_cost = latent_cost_model.predict_f(x2);
    u0_x2_latent_cost = u0_x2_latent_cost.numpy();
    var0_x2_latent_cost = var0_x2_latent_cost.numpy();
    u0_x2_cost = np.exp(u0_x2_latent_cost)

    return u0_x2_cost, u0_x2_latent_cost, var0_x2_latent_cost

def get_lower_cost_bound(u_x_lat_cost, var_x_lat_cost):

    return np.exp(u_x_lat_cost- 1.96*np.sqrt(var_x_lat_cost))

def get_upper_cost_bound(u_x_lat_cost, var_x_lat_cost):

    return np.exp(u_x_lat_cost+ 1.96* np.sqrt(var_x_lat_cost))

def get_dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise):

    du_dx, dvar_dx, _= get_mean_variance_gradients_q1(kernel, Xt, Yt, x, noise)

    dsigma_dx= (1/(2*sigma_x))*dvar_dx

    '''cumulative gradient'''
    dcumulative_dx= dcumulative_du(f_best, sigma_x,u_x)*du_dx + \
                            dcumulative_dsigma(f_best, sigma_x, u_x)*dsigma_dx

    term1= (f_best- u_x)*dcumulative_dx

    term2= -du_dx* norm.cdf(((f_best- u_x)/sigma_x))

    term3= dsigma_dx* norm.pdf((f_best- u_x)/sigma_x)

    '''normal distribution gradient'''

    dnormal_dx= dnormal_du(f_best, u_x, sigma_x)*du_dx+ dnormal_dsigma(f_best, u_x, sigma_x)*dsigma_dx

    term4= sigma_x* dnormal_dx

    return term1+ term2+ term3+ term4



def ei_lapu_optimize_grid(X2_grid, latent_cost_model, model, f_best):


    u_X2_grid, var_X2_grid, sigma_X2_grid= get_mean_var_std(X2_grid, model)

    u_cost_X2_grid, u_lat_cost_X2_grid, var_lat_cost_X2_grid= get_mean_and_var_cost(X2_grid, latent_cost_model)
    # cost_lower_X2_grid= get_lower_cost_bound(u_lat_cost_X2_grid, var_lat_cost_X2_grid)

    EI_X2_grid= ei(sigma_X2_grid, u_X2_grid, f_best)

    # EI1_cepu_X2_grid= EI1_X2_grid/cost_lower_X2_grid- get_worst_scenario_cost_loss(latent_cost_model, cost_lower_X2_grid,
    #                                                                                Yt_latent)
    EI_X2_cepu_grid= EI_X2_grid/u_cost_X2_grid

    index_max_grid= int(np.argmax(EI_X2_cepu_grid, axis=0))
    x2_opt= X2_grid[index_max_grid, :].reshape(1,-1)
    x2_opt_value= EI_X2_cepu_grid[index_max_grid, :].reshape(1,-1)

    '''EI1_x2_opt'''
    ei_x2_opt= EI_X2_grid[index_max_grid, :].reshape(1,-1)
    # print('u_x2_opt_lat_cost:{}, var_x2_opt_lat_cost:{} found by la cost model'.
    #       format(u_lat_cost_X2_grid[index_max_grid, :].reshape(1,-1), var_lat_cost_X2_grid[index_max_grid, :].reshape(1,-1)))
    # print('cost_lower_x2_opt_lat_cost:{} by la model'.format(cost_lower_X2_grid[index_max_grid, :].reshape(1,-1)))
    return x2_opt, x2_opt_value, ei_x2_opt


'''inner optimization'''
class Ei_pu_la_optimize:

    def __init__(self):
        pass

    def ei_pu_x2_negative(self, x2, latent_cost_model_la,latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best, Xt, Yt, noise):

        x2= x2.reshape(1,-1)
        self.u_x2, self.var_x2, self.sigma_x2= get_mean_var_std(x2, model)
        self.u_x2_cost, self.u_x2_lat_cost, self.var_x2_lat_cost= get_mean_and_var_cost(x2, latent_cost_model_la)
        self.sigma_x2_lat_cost= np.sqrt(self.var_x2_lat_cost)

        # self.cost_low_x2 = get_lower_cost_bound(self.u_x2_lat_cost, self.var_x2_lat_cost)

        self.ei_x2=ei(self.sigma_x2, self.u_x2, f_best)
        self.ei_pu_x2= self.ei_x2/ self.u_x2_cost
        self.ei_pu_x2= self.ei_pu_x2.flatten()

        return -self.ei_pu_x2

    def grad_ei_pu_x2_negative(self, x2, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best, Xt, Yt, noise):

        x2 = x2.reshape(1, -1)
        #grad denom
        du_lat_cost_dx2, dvar_lat_cost_dx2, _= get_mean_variance_gradients_q1(latent_cost_kernel, Xt1, Yt1_latent_cost, x2, noise_cost)
        # dsigma_lat_cost_dx2= (1/(2*self.sigma_x2_lat_cost))*dvar_lat_cost_dx2
        # grad_cost_low= (du_lat_cost_dx2 - 1.96* dsigma_lat_cost_dx2)*self.cost_low_x2
        # grad_cost_low= (du_lat_cost_dx2)*self.u_x2_cost
        grad_cost= (du_lat_cost_dx2)*self.u_x2_cost

        #grad nom
        grad_ei =get_dEI_dx(f_best, self.u_x2, self.sigma_x2, kernel, Xt, Yt, x2, noise)

        #grad_ei_pu
        grad_ei_pu= (self.u_x2_cost*grad_ei- grad_cost*self.ei_x2)/self.u_x2_cost**2
        grad_ei_pu= grad_ei_pu.flatten()
        return -grad_ei_pu

    def maximize_ei_pu(self, kernel, latent_cost_model_la, Xt, noise, model, domain, num_inner_opt_restarts, f_best,
                       grid, D, num_iter_max, noise_cost, latent_cost_kernel, Xt1, Yt1_latent_cost, Yt):

        '''scipy optimize'''
        if num_inner_opt_restarts > 0:
            lower = [];
            upper = []
            for i in range(len(domain)):
                lower.append(domain[i][0])
                upper.append(domain[i][1])
            b = Bounds(lb=lower, ub=upper)

            x2_opt_list_sci = np.zeros([num_inner_opt_restarts, D])
            x2_opt_value_list_sci = np.zeros([num_inner_opt_restarts, 1])

            for i in range(num_inner_opt_restarts):
                x2 = np.random.uniform(lower, upper, (1, D))
                fun_args = (latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best, Xt, Yt, noise)

                result = (minimize(self.ei_pu_x2_negative, x2, args=fun_args, bounds=b, method='L-BFGS-B',
                                   jac=self.grad_ei_pu_x2_negative, options={'maxiter': num_iter_max}))

                x2_cand = result['x'];
                x2_cand = x2_cand.reshape(1, -1)
                x2_cand_value = -result['fun'];
                x2_cand_value = x2_cand_value.reshape(1, -1)
                x2_cand_grad = -result['jac'];
                x2_cand_grad = x2_cand_grad.reshape(1, -1)

                x2_opt_list_sci[i, :] = x2_cand[0, :];
                x2_opt_value_list_sci[i, :] = x2_cand_value

            index_opt_sci = int(np.argmax(x2_opt_value_list_sci, axis=0))

            x2_opt_value_sci = x2_opt_value_list_sci[index_opt_sci, :].reshape(1, -1)

            x2_opt_sci = x2_opt_list_sci[index_opt_sci, :].reshape(1, -1)

            u_x2_opt_sci, var_x2_opt_sci = model.predict_f(x2_opt_sci);
            u_x2_opt_sci = u_x2_opt_sci.numpy();
            var_x2_opt_sci = var_x2_opt_sci.numpy()
            sigma_x2_opt_sci= np.sqrt(var_x2_opt_sci)

            ei_x2_opt_sci = ei(sigma_x2_opt_sci, u_x2_opt_sci, f_best)

        '''grid optimize'''

        '''inner optimization with grid'''
        if D == 1 and grid:
            disc = 101
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            X2_grid = x1_grid.reshape(-1, 1)

            x2_opt_grid, x2_opt_value_grid, ei_x2_opt_grid = ei_lapu_optimize_grid(X2_grid, latent_cost_model_la, model, f_best)


        if D == 2 and grid:
            disc = 21
            x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
            x2_grid = np.linspace(domain[1][0], domain[1][1], disc)
            # x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)
            X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

            X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
            X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
            X2_grid = np.append(X1_flat, X2_flat, axis=1)

            x2_opt_grid, x2_opt_value_grid, ei_x2_opt_grid = ei_lapu_optimize_grid(X2_grid, latent_cost_model_la, model, f_best)
        '''compare grid with sci optimum'''

        if (D == 2 or D == 1) and grid and (num_inner_opt_restarts > 0):
            if x2_opt_value_grid > x2_opt_value_sci:
                x2_opt = x2_opt_grid;
                x2_opt_value = x2_opt_value_grid
                ei_x2_opt = ei_x2_opt_grid
            else:
                x2_opt = x2_opt_sci;
                x2_opt_value = x2_opt_value_sci
                ei_x2_opt = ei_x2_opt_sci
        elif num_inner_opt_restarts > 0:
            x2_opt = x2_opt_sci;
            x2_opt_value = x2_opt_value_sci
            ei_x2_opt = ei_x2_opt_sci
        else:
            x2_opt = x2_opt_grid
            x2_opt_value = x2_opt_value_grid
            ei_x2_opt = ei_x2_opt_grid

        return x2_opt_value, x2_opt, ei_x2_opt


def get_worst_case_ei_pu_loss(kernel, latent_cost_kernel, noise_cost, x, Xt, Yt_latent_cost,
                              noise, model, domain, f_best, grid, D, num_iter_max, num_opt_restarts, u_x, sigma_x, cost_upper_x, Yt):

    #calculate ei_pu_x for the worst cost
    # cost_upper_x= get_upper_cost_bound(u_x_lat_cost, var_x_lat_cost)
    ei_pu_x_worst = ei_pu(sigma_x, u_x, f_best, cost_upper_x)

    #assume worst case cost
    Xt1= np.append(Xt, x, axis= 0)
    Yt1_latent_cost= np.append(Yt_latent_cost, np.log(cost_upper_x), axis=0)

    #update model accordingly
    latent_cost_model_la= gp.models.GPR((Xt1, Yt1_latent_cost), latent_cost_kernel)
    latent_cost_model_la.likelihood.variance.assign(noise_cost)

    #calculate optimum ei_pu in the worst case
    opt= Ei_pu_la_optimize()
    x2_opt_ei_pu, x2_opt, ei_x2_opt = opt.maximize_ei_pu(kernel, latent_cost_model_la, Xt, noise, model, domain, num_opt_restarts, f_best,
                       grid, D, num_iter_max, noise_cost, latent_cost_kernel, Xt1, Yt1_latent_cost, Yt)

    #loss of this worst outcome
    loss=  x2_opt_ei_pu- ei_pu_x_worst

    return loss, x2_opt, ei_x2_opt, x2_opt_ei_pu, ei_pu_x_worst, latent_cost_model_la


def get_posterior_covariance_and_derivative_1q(x2, x, Xt, kernel, noise):

    #K_x1_x= K_x_x1

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    k_x_Xt= (kernel.K(x,Xt)).numpy()
    k_x2_Xt_T= np.transpose((kernel.K(x2, Xt)).numpy())
    alpha= np.matmul(K_inv, k_x2_Xt_T)
    k_x_x2= (kernel.K(x,x2)).numpy()
    K_x_x2= (k_x_x2- np.matmul(k_x_Xt, alpha))


    '''gradient for 1d'''
    ls = (kernel.lengthscales._value()).numpy()

    if type(ls)== np.ndarray:

        lamb= np.diagflat(ls)
        lamb_inv= np.diagflat(1/ls)
    else:

        lamb= np.eye(Xt.shape[1])*ls
        lamb_inv= np.eye(Xt.shape[1])*1/ls

    k_x_x2= (kernel.K(x,x2)).numpy()
    diff_vec= (x2-x).T
    term1= np.matmul(lamb_inv, (diff_vec* k_x_x2))
    term2 = -np.matmul(lamb_inv, np.matmul(np.transpose(x-Xt)*k_x_Xt, alpha))
    dK_x_x2_dx= term1-term2

    return K_x_x2, dK_x_x2_dx

def funfi(A):

    result= np.tril(A, k=-1)+ 1/2*np.diagflat(np.diag(A))
    return result

def get_one_step_mean_and_variance(u0_x2_lat_cost, var0_x2_lat_cost, sigma0):

    u1_x2_lat_cost= u0_x2_lat_cost + np.matmul(sigma0,np.atleast_2d(1.96))

    var1_x2_lat_cost= var0_x2_lat_cost- np.matmul(sigma0, sigma0.T)


    return u1_x2_lat_cost, var1_x2_lat_cost

def get_mean_var_gradient_of_la_term(dsigma0_dx, sigma0, D):
    du_x2_dx = np.matmul(dsigma0_dx, np.atleast_2d(1.96))

    dvar_x2_dx = np.empty([D, 1])

    '''holds also for q>1'''
    for i in range(D):
        dsigma0_dxi = dsigma0_dx[i, :].reshape(-1, 1)
        dvar_x2_dxi = -2 * np.matmul(sigma0, np.transpose(dsigma0_dxi))
        dvar_x2_dx[i, 0] = dvar_x2_dxi

    return du_x2_dx, dvar_x2_dx

def get_dsigma0_dx_q1(x2, x, Xt, latent_cost_kernel, noise_cost, dK0_x_dx, D, L0_x, L0_x_inv, L0_x_inv_T):

    #K0_x1: 1*1, dK0_x1_dx1: D*1
    K0_x2_x, dK0_x2_x_dx = get_posterior_covariance_and_derivative_1q(x2, x, Xt, latent_cost_kernel, noise_cost)
    # _, dK0_x1_dx1, K0_x1_x1= mean_variance_gradients(kernel, Xt, Yt, x1, noise)

    # L0_x1= np.linalg.cholesky(K0_x1_x1)
    # L0_x1_inv= np.linalg.inv(L0_x1); L0_x1_inv_T= np.transpose(L0_x1_inv)

    dL0_x_dx = np.empty([D, 1])

    for i in range(D):
        dK0_x_dxi = dK0_x_dx[i, :].reshape(1, 1)
        Ai = np.matmul(L0_x_inv, np.matmul(dK0_x_dxi, L0_x_inv_T))
        temp = funfi(Ai)
        dL0_x_dxi = np.matmul(L0_x, temp)
        dL0_x_dx[i, 0] = dL0_x_dxi

    '''only for q=1'''
    dL0_x_inv_dx=  np.empty([D,1])
    for i in range(D):
        dL0_x_dxi= dL0_x_dx[i,:].reshape(1,1)
        dL0_x_inv_dxi= -np.matmul(L0_x_inv, np.matmul(dL0_x_dxi, L0_x_inv))
        dL0_x_inv_dx[i,0]= dL0_x_inv_dxi

    '''only for q=1'''
    dsigma0_dx= np.empty([D,1])
    for i in range(D):
        dK0_x2_x_dxi= dK0_x2_x_dx[i,:].reshape(1,1);
        dL0_x_inv_dxi= dL0_x_inv_dx[i,:].reshape(1,1)
        dsigma0_dxi= np.matmul(np.transpose(L0_x_inv),dK0_x2_x_dxi)+ np.matmul(K0_x2_x, dL0_x_inv_dxi)
        dsigma0_dx[i,0]= dsigma0_dxi

    sigma0= np.matmul(K0_x2_x, L0_x_inv)

    return dsigma0_dx, sigma0, dL0_x_dx

class Ei_cepu_v2_optimize:


    def __init__(self):

        pass

    def ei_cepu_negative(self, x, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D):


        # print('function')
        x = x.reshape(1, -1)

        self.u_x_cost, self.u_x_lat_cost, self.var_x_lat_cost = get_mean_and_var_cost(x, latent_cost_model)
        self.cost_lower_x = get_lower_cost_bound(self.u_x_lat_cost, self.var_x_lat_cost)
        self.cost_upper_x = get_upper_cost_bound(self.u_x_lat_cost, self.var_x_lat_cost)

        self.u_x, self.var_x, self.sigma_x= get_mean_var_std(x, model)

        #calculate loss in the worst case

        ei_pu_loss, self.x2_opt, self.ei_x2_opt, self.x2_opt_ei_pu, self.ei_pu_x_worst, self.latent_cost_model_la=\
                                                        get_worst_case_ei_pu_loss(kernel, latent_cost_kernel, noise_cost, x, Xt, Yt_latent_cost,
                                                                noise, model, domain, f_best, grid_opt_in, D, num_iter_max,
                                                              num_inner_opt_restarts, self.u_x, self.sigma_x, self.cost_upper_x, Yt)

        EI_x_cepu =  ei_pu(self.sigma_x, self.u_x , f_best, self.cost_lower_x) - ei_pu_loss

        EI_x_cepu = EI_x_cepu.flatten()

        return -EI_x_cepu#, ei_pu(self.sigma_x, self.u_x , f_best, self.cost_lower_x), self.ei_pu_x_worst, self.x2_opt_ei_pu


    def grad_ei_cepu_negative(self, x, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D):


        # print('gradient')
        x = x.reshape(1, -1)

        '''dEI_x'''

        EI_x_gradient = get_dEI_dx(f_best, self.u_x, self.sigma_x, kernel, Xt, Yt, x, noise)

        '''gradient of lower and upper cost of x'''

        self.du_x_lat_cost, self.dvar_x_lat_cost, K_x_lat_cost = get_mean_variance_gradients_q1(latent_cost_kernel, Xt, Yt_latent_cost, x, noise_cost)

        sigma_x_lat_cost = np.sqrt(self.var_x_lat_cost)

        self.dsigma_x_lat_cost = (1/(2* sigma_x_lat_cost))*self.dvar_x_lat_cost

        self.grad_lower_cost_x= self.cost_lower_x*(self.du_x_lat_cost- 1.96*self.dsigma_x_lat_cost)
        self.grad_upper_cost_x= self.cost_upper_x*(self.du_x_lat_cost+ 1.96*self.dsigma_x_lat_cost)

        '''gradient of second step lower-cost'''
        # du_lat_dx_cost, dvar_lat_dx_cost, K_x_lat_cost= get_mean_variance_gradients_q1(latent_cost_kernel, Xt, Yt_latent_cost, x, noise)
        L0_x_lat_cost = np.linalg.cholesky(K_x_lat_cost)
        L0_x_lat_cost_inv = np.linalg.inv(L0_x_lat_cost);
        L0_x_lat_cost_inv_T = np.transpose(L0_x_lat_cost_inv)

        dsigma0_dx_cost, sigma0_cost, dL0_x_dx_cost= get_dsigma0_dx_q1(self.x2_opt, x, Xt, latent_cost_kernel, noise_cost, self.dvar_x_lat_cost, D,
                                                       L0_x_lat_cost, L0_x_lat_cost_inv, L0_x_lat_cost_inv_T)

        du_lat_x2_dx_cost, dvar_lat_x2_dx_cost= get_mean_var_gradient_of_la_term(dsigma0_dx_cost, sigma0_cost, D)

        u0_x2_cost, u0_x2_lat_cost, var0_x2_lat_cost = get_mean_and_var_cost(self.x2_opt, latent_cost_model)
        self.u1_lat_cost_x2, self.var1_lat_cost_x2= get_one_step_mean_and_variance(u0_x2_lat_cost, var0_x2_lat_cost, sigma0_cost)
        sigma1_lat_cost_x2= np.sqrt(self.var1_lat_cost_x2)
        dsigma_lat_cost_x2= (1/(2*sigma1_lat_cost_x2))*dvar_lat_x2_dx_cost
        self.cost1_lower_x2= np.exp(self.u1_lat_cost_x2- 1.96*sigma1_lat_cost_x2)

        self.dcost1_lower_x2_dx= (du_lat_x2_dx_cost- 1.96*dsigma_lat_cost_x2)*self.cost1_lower_x2

        '''ei_x'''
        ei_x= ei(self.sigma_x, self.u_x, f_best)

        '''overall gradient term'''
        term1_grad= (self.cost_lower_x*EI_x_gradient- ei_x* self.grad_lower_cost_x)/(self.cost_lower_x**2)
        term2_grad= (self.cost_upper_x*EI_x_gradient- ei_x* self.grad_upper_cost_x)/(self.cost_upper_x**2)
        term3_grad= (self.dcost1_lower_x2_dx*self.ei_x2_opt)/(self.cost1_lower_x2**2)

        result= term1_grad+term2_grad+term3_grad
        result= result.flatten()

        return -result

    def maximize_ei_cepu(self, kernel, latent_cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
                             num_inner_opt_restarts, num_outer_opt_restarts, grid_opt_in, grid_opt_out, D, Yt_latent_cost, num_iter_max, noise_cost):

        '''scipy optimize'''
        if num_outer_opt_restarts > 0:
            lower = [];
            upper = []
            for i in range(len(domain)):
                lower.append(domain[i][0])
                upper.append(domain[i][1])
            b = Bounds(lb=lower, ub=upper)

            x_opt_list_sci = np.zeros([num_outer_opt_restarts, D])
            x_opt_value_list_sci = np.zeros([num_outer_opt_restarts, 1])

            for i in range(num_outer_opt_restarts):
                x0 = np.random.uniform(lower, upper, (1, D))
                print('outer_opt:{}'.format(i))
                fun_args= (model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)

                result = (minimize(self.ei_cepu_negative, x0, args= fun_args, bounds=b, method='L-BFGS-B', jac= self.grad_ei_cepu_negative,
                          options={'maxiter':num_iter_max}))

                x_cand = result['x'];
                x_cand = x_cand.reshape(1, -1)
                x_cand_value = -result['fun'];
                x_cand_value = x_cand_value.reshape(1, -1)
                x_cand_grad = -result['jac'];
                x_cand_grad = x_cand_grad.reshape(1, -1)

                x_opt_list_sci[i, :] = x_cand[0, :];
                x_opt_value_list_sci[i, :] = x_cand_value

            index_opt_sci = int(np.argmax(x_opt_value_list_sci, axis=0))

            x_opt_value_sci = x_opt_value_list_sci[index_opt_sci, :].reshape(1, -1)

            x_opt_sci = x_opt_list_sci[index_opt_sci, :].reshape(1, -1)

            # u_x_opt_sci, var_x_opt_sci, sigma_x_opt_sci = get_mean_var_std(x_opt_sci, model)

            # ei_x_opt_sci = ei(sigma_x_opt_sci,  u_x_opt_sci, f_best)

        '''grid optimize'''

        '''outer optimization with grid'''
        if grid_opt_out== True:
            if D==1:

                disc = 101
                x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
                X_grid = x1_grid.reshape(-1, 1)

                x_value_list_grid= np.zeros([disc, 1])

                for j in range(disc):
                    x_cand= X_grid[j,:].reshape(1,-1)

                    x_cand_value= -self.ei_cepu_negative(x_cand, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                            noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D);

                    x_cand_value= x_cand_value.reshape(1,-1)
                    x_value_list_grid[j, :]= x_cand_value[0,:]

                index_max_grid = int(np.argmax(x_value_list_grid, axis=0))
                x_opt_value_grid =x_value_list_grid[index_max_grid, :].reshape(1, -1)
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

                x_value_list_grid = np.zeros([disc**2, 1])


                for j in range(disc**2):
                    x_cand = X_grid[j, :].reshape(1, -1)
                    x_cand_value =-self.ei_cepu_negative(x_cand, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                            noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D);

                    x_cand_value = x_cand_value.reshape(1, -1)
                    x_value_list_grid[j, :] = x_cand_value[0, :]

                index_max_grid = int(np.argmax(x_value_list_grid, axis=0))
                x_opt_value_grid =x_value_list_grid[index_max_grid, :].reshape(1, -1)
                x_opt_grid = X_grid[index_max_grid, :].reshape(1, -1)

        '''compare grid with sci optimum'''

        if (D == 2 or D == 1) and grid_opt_out and (num_outer_opt_restarts > 0):
            if x_opt_value_grid > x_opt_value_sci:
                x_opt = x_opt_grid;
                x_opt_value = x_opt_value_grid
            else:
                x_opt = x_opt_sci;
                x_opt_value = x_opt_value_sci
            return x_opt, x_opt_value, x_value_list_grid, X_grid

        elif (not grid_opt_out) and (num_outer_opt_restarts > 0):
            x_opt = x_opt_sci;
            x_opt_value = x_opt_value_sci

            return x_opt, x_opt_value, None, None

        elif grid_opt_out and (not num_outer_opt_restarts > 0):
            x_opt = x_opt_grid
            x_opt_value = x_opt_value_grid

            return x_opt, x_opt_value, x_value_list_grid, X_grid


import sys
sys.path.append('./gp_gradients')
sys.path.append('..')

from gp_gradients import EI1_x2_per_cost_optimize

sys.path.append('../functions')
from sine import *
sys.path.append('../cost_functions')
from cos_1d import *
from exp_cos_1d import *

import time


sys.path.append('../functions')
sys.path.append('../cost_functions')
from exp_cos_2d import *
from branin import *

def temp_test_ei_cepu_opt_2d():

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


    x1_test = np.random.uniform(lower, upper, (1, D))

    opt= Ei_cepu_optimize()
    # opt.maximize_ei_cepu(kernel, latent_cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
    #                          num_inner_opt_restarts, num_outer_opt_restarts, grid_opt_in, grid_opt_out, D, Yt_latent_cost, num_iter_max, noise_cost)

    x1_test_val, term1_test, term2_test, term3_test= opt.ei_cepu_negative(x1_test, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)
    u_x_test_lat_cost= opt.u_x_lat_cost; var_x_test_lat_cost= opt.var_x_lat_cost

    x1_test_val*=-1
    '''analytical gradient'''
    x1_test_grad, term1_test_grad, term2_test_grad, term3_test_grad= opt.grad_ei_cepu_negative(x1_test, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)

    du_x_test_lat_cost = opt.du_x_lat_cost; dvar_x_test_lat_cost = opt.dvar_x_lat_cost
    cost_lower_x_test= opt.cost_lower_x; cost_upper_x_test= opt.cost_upper_x
    grad_lower_cost_x_test= opt.grad_lower_cost_x; grad_upper_cost_x_test= opt.grad_upper_cost_x
    dcost1_lower_x2_dx_test = opt.dcost1_lower_x2_dx
    cost1_lower_x2_test = opt.cost1_lower_x2

    cost1_lower_x_test= opt.cost1_lower_x2; grad_cost1_lower_x_test= opt.dcost1_lower_x2_dx
    print('cost1_lower_x2:{} in grad'.format(opt.cost1_lower_x2))
    print('u1_x2_opt_lat_cost_code:{}, var1_x2_opt_lat_cost_code:{}'.format(opt.u1_lat_cost_x2, opt.var1_lat_cost_x2))
    print('cost_lower_x_test:{}, cost_lower_x_test_in_grad:{}'.format(opt.ei_x2_opt/opt.x2_opt_ei_pu, cost_lower_x_test))
    x1_test_grad*=-1
    '''approximate gradient'''

    x11= x1_test.copy(); x11[0,0]= x11[0,0]+ 0.000001;
    x12= x1_test.copy(); x12[0,0] = x12[0,0]- 0.000001
    x21= x1_test.copy(); x21[0,1] = x21[0,1]+ 0.000001
    x22= x1_test.copy(); x22[0,1] = x22[0,1]- 0.000001


    x11_val, term1_11, term2_11, term3_11=   opt.ei_cepu_negative(x11, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)
    u_x_11_lat_cost= opt.u_x_lat_cost; var_x11_lat_cost= opt.var_x_lat_cost
    cost_lower_x11= opt.cost_lower_x; cost_upper_x11= opt.cost_upper_x
    cost1_lower_x2_x11 = opt.ei_x2_opt/opt.x2_opt_ei_pu

    x12_val, term1_12, term2_12, term3_12=   opt.ei_cepu_negative(x12, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)
    u_x_12_lat_cost= opt.u_x_lat_cost; var_x12_lat_cost= opt.var_x_lat_cost
    cost_lower_x12= opt.cost_lower_x; cost_upper_x12= opt.cost_upper_x
    cost1_lower_x2_x12 = opt.ei_x2_opt/opt.x2_opt_ei_pu

    x21_val, term1_21, term2_21, term3_21=   opt.ei_cepu_negative(x21, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)
    u_x_21_lat_cost= opt.u_x_lat_cost; var_x21_lat_cost= opt.var_x_lat_cost
    cost_lower_x21= opt.cost_lower_x; cost_upper_x21=  opt.cost_upper_x
    cost1_lower_x2_x21 =opt.ei_x2_opt/opt.x2_opt_ei_pu

    x22_val, term1_22, term2_22, term3_22=   opt.ei_cepu_negative(x22, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)
    u_x_22_lat_cost= opt.u_x_lat_cost; var_x22_lat_cost= opt.var_x_lat_cost
    cost_lower_x22= opt.cost_lower_x; cost_upper_x22= opt.cost_upper_x
    cost1_lower_x2_x22 = opt.ei_x2_opt/opt.x2_opt_ei_pu

    x11_val*=-1
    x12_val*=-1
    x21_val*=-1
    x22_val*=-1
    grad_app_dir1 = ((x11_val -x1_test_val) / (x11[0, 0] - x1_test[0, 0]) + (x1_test_val - x12_val) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2 = ((x21_val - x1_test_val) / (x21[0, 1] - x1_test[0, 1]) + (x1_test_val - x22_val) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_dir1_term1 = ((term1_11- term1_test) / (x11[0, 0] - x1_test[0, 0]) + (term1_test - term1_12) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2_term1 = ((term1_21- term1_test) / (x21[0, 1] - x1_test[0, 1]) + (term1_test - term1_22) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_dir1_term2 = ((term2_11- term2_test) / (x11[0, 0] - x1_test[0, 0]) + (term2_test - term2_12) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2_term2 = ((term2_21- term2_test) / (x21[0, 1] - x1_test[0, 1]) + (term2_test - term2_22) / (x1_test[0, 1] - x22[0, 1])) / 2;

    # term3_test= term23_test- term2_test
    # term3_11= term23_11-term2_11;
    # term3_12= term23_12-term2_12;
    # term3_21= term23_21-term2_21;
    # term3_22= term23_22-term2_22;

    grad_app_dir1_term3 = ((term3_11- term3_test) / (x11[0, 0] - x1_test[0, 0]) + (term3_test - term3_12) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_dir2_term3 = ((term3_21- term3_test) / (x21[0, 1] - x1_test[0, 1]) + (term3_test - term3_22) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_u_x_lat_cost_dir1 = ((u_x_11_lat_cost- u_x_test_lat_cost) / (x11[0, 0] - x1_test[0, 0]) + (u_x_test_lat_cost - u_x_12_lat_cost) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_u_x_lat_cost_dir2 = ((u_x_21_lat_cost-u_x_test_lat_cost) / (x21[0, 1] - x1_test[0, 1]) + (u_x_test_lat_cost - u_x_22_lat_cost) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_var_x_lat_cost_dir1 = ((var_x11_lat_cost- var_x_test_lat_cost) / (x11[0, 0] - x1_test[0, 0]) + (var_x_test_lat_cost - var_x12_lat_cost) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_var_x_lat_cost_dir2 = ((var_x21_lat_cost- var_x_test_lat_cost) / (x21[0, 1] - x1_test[0, 1]) + (var_x_test_lat_cost - var_x22_lat_cost) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_lower_cost_dir1 = ((cost_lower_x11- cost_lower_x_test) / (x11[0, 0] - x1_test[0, 0]) + (cost_lower_x_test - cost_lower_x12) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_lower_cost_dir2 = ((cost_lower_x21- cost_lower_x_test) / (x21[0, 1] - x1_test[0, 1]) + (cost_lower_x_test - cost_lower_x22) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_upper_cost_dir1 = ((cost_upper_x11- cost_upper_x_test) / (x11[0, 0] - x1_test[0, 0]) + (cost_upper_x_test - cost_upper_x12) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_upper_cost_dir2 = ((cost_upper_x21- cost_upper_x_test) / (x21[0, 1] - x1_test[0, 1]) + (cost_upper_x_test - cost_upper_x22) / (x1_test[0, 1] - x22[0, 1])) / 2;

    grad_app_cost1_lower_dir1 = ((cost1_lower_x2_x11- cost1_lower_x2_test) / (x11[0, 0] - x1_test[0, 0]) + (cost1_lower_x2_test - cost1_lower_x2_x12) / (x1_test[0, 0] - x12[0, 0])) / 2;
    grad_app_cost1_lower_dir2 = ((cost1_lower_x2_x21- cost1_lower_x2_test) / (x21[0, 1] - x1_test[0, 1]) + (cost1_lower_x2_test - cost1_lower_x2_x22) / (x1_test[0, 1] - x22[0, 1])) / 2;


    grad_app = np.append(grad_app_dir1, grad_app_dir2, axis=0)
    grad_app_term1 = np.append(grad_app_dir1_term1, grad_app_dir2_term1, axis=0)
    grad_app_term2 = np.append(grad_app_dir1_term2, grad_app_dir2_term2, axis=0)
    grad_app_term3 = np.append(grad_app_dir1_term3, grad_app_dir2_term3, axis=0)
    grad_app_u_x_lat_cost= np.append(grad_app_u_x_lat_cost_dir1, grad_app_u_x_lat_cost_dir2, axis=0)
    grad_app_var_x_lat_cost= np.append(grad_app_var_x_lat_cost_dir1, grad_app_var_x_lat_cost_dir2, axis=0)
    grad_app_cost_lower= np.append(grad_app_lower_cost_dir1, grad_app_lower_cost_dir2, axis=0)
    grad_app_cost_upper= np.append(grad_app_upper_cost_dir1, grad_app_upper_cost_dir2, axis=0)
    grad_app_cost1_lower= np.append(grad_app_cost1_lower_dir1, grad_app_cost1_lower_dir2, axis=0)

    print('analytical_grad: \n{}, \napp_grad:\n{}'.format(x1_test_grad, grad_app))
    print('analytical_grad_term1: \n{}, \napp_grad_term1:\n{}'.format(term1_test_grad, grad_app_term1))
    print('analytical_grad_term2: \n{}, \napp_grad_term2:\n{}'.format(term2_test_grad, grad_app_term2))
    print('analytical_grad_term3: \n{}, \napp_grad_term3:\n{}'.format(term3_test_grad, grad_app_term3))
    print('grad_u_x_lat_cost_analytical: \n{}, \napp_grad_u_x_lat_cost:\n{}'.format(du_x_test_lat_cost, grad_app_u_x_lat_cost))
    print('grad_var_x_lat_cost_analytical: \n{}, \napp_grad_var_x_lat_cost:\n{}'.format(dvar_x_test_lat_cost, grad_app_var_x_lat_cost))
    print('grad_lower_cost_analytical: \n{}, \napp_grad_lower_cost:\n{}'.format(grad_lower_cost_x_test, grad_app_cost_lower))
    print('grad_upper_cost_analytical: \n{}, \napp_grad_app_upper_cost:\n{}'.format(grad_upper_cost_x_test, grad_app_cost_upper))
    print('grad_cost1_lower_analytical: \n{}, \napp_grad_app_lower_cost:\n{}'.format(dcost1_lower_x2_dx_test, grad_app_cost1_lower))

    x1_test = np.random.uniform(lower, upper, (1, D))

    Xt1= np.append(Xt, x1_test, axis= 0)
    u_x_cost_test, u_x_lat_cost_test, var_x_lat_cost_test = get_mean_and_var_cost(x1_test, latent_cost_model)
    cost_upper_x_test = get_upper_cost_bound(u_x_lat_cost_test, var_x_lat_cost_test)

    Yt1_latent_cost= np.append(Yt_latent_cost, np.log(cost_upper_x_test), axis=0)
    x2_test = np.random.uniform(lower, upper, (1, D))

    #update model accordingly
    latent_cost_model_la= gp.models.GPR((Xt1, Yt1_latent_cost), latent_cost_kernel)
    latent_cost_model_la.likelihood.variance.assign(noise_cost)
    inner_opt= Ei_pu_la_optimize()

    ei_pu_x2_test, cost_low_x2_test= inner_opt.ei_pu_x2_negative(x2_test, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best,
                                Xt, Yt, noise)
    grad_ei_pu_x2_test, grad_cost_low_x2_test= inner_opt.grad_ei_pu_x2_negative(x2_test, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best,
                                Xt, Yt, noise)
    ei_pu_x2_test*= -1; grad_ei_pu_x2_test*=-1

    x2_11= x2_test.copy(); x2_11[0,0]= x2_11[0,0]+ 0.000001;
    x2_12= x2_test.copy(); x2_12[0,0] = x2_12[0,0]- 0.000001
    x2_21= x2_test.copy(); x2_21[0,1] = x2_21[0,1]+ 0.000001
    x2_22= x2_test.copy(); x2_22[0,1] = x2_22[0,1]- 0.000001

    ei_pu_x2_11, cost_low_x2_11= inner_opt.ei_pu_x2_negative(x2_11, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best,
                                Xt, Yt, noise)
    ei_pu_x2_11 *= -1;
    ei_pu_x2_12, cost_low_x2_12= inner_opt.ei_pu_x2_negative(x2_12, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best,
                                Xt, Yt, noise)
    ei_pu_x2_12 *= -1;

    ei_pu_x2_21, cost_low_x2_21= inner_opt.ei_pu_x2_negative(x2_21, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best,
                                Xt, Yt, noise)
    ei_pu_x2_21 *= -1;

    ei_pu_x2_22, cost_low_x2_22= inner_opt.ei_pu_x2_negative(x2_22, latent_cost_model_la, latent_cost_kernel, model, kernel, Xt1, Yt1_latent_cost, noise_cost, f_best,
                                Xt, Yt, noise)
    ei_pu_x2_22 *= -1;

    grad_app_x2_dir1 = ((ei_pu_x2_11 -ei_pu_x2_test) / (x2_11[0, 0] - x2_test[0, 0]) + (ei_pu_x2_test - ei_pu_x2_12) / (x2_test[0, 0] - x2_12[0, 0])) / 2;
    grad_app_x2_dir2 = ((ei_pu_x2_21 - ei_pu_x2_test) / (x2_21[0, 1] - x2_test[0, 1]) + (ei_pu_x2_test - ei_pu_x2_22) / (x2_test[0, 1] - x2_22[0, 1])) / 2;

    grad_app_cost_low_x2_dir1 = ((cost_low_x2_11 -cost_low_x2_test) / (x2_11[0, 0] - x2_test[0, 0]) + (cost_low_x2_test - cost_low_x2_12) / (x2_test[0, 0] - x2_12[0, 0])) / 2;
    grad_app_cost_low_x2_dir2 = ((cost_low_x2_21 - cost_low_x2_test) / (x2_21[0, 1] - x2_test[0, 1]) + (cost_low_x2_test - cost_low_x2_22) / (x2_test[0, 1] - x2_22[0, 1])) / 2;

    grad_app_x2 = np.append(grad_app_x2_dir1, grad_app_x2_dir2, axis=0)
    grad_app_cost_low_x2_test = np.append(grad_app_cost_low_x2_dir1, grad_app_cost_low_x2_dir2, axis=0)

    print('analytical_grad: \n{}, \napp_grad:\n{}'.format(grad_ei_pu_x2_test, grad_app_x2))
    print('analytical_grad_cost_lower: \n{}, \napp_grad_cost_lower:\n{}'.format(grad_cost_low_x2_test, grad_app_cost_low_x2_test))

    def calculate_lower_cost(x2, model, latent_cost_model_la):
        u_x2_cost, u_x2_lat_cost, var_x2_lat_cost = get_mean_and_var_cost(x2, latent_cost_model_la)

        cost_low_x2 = get_lower_cost_bound(u_x2_lat_cost, var_x2_lat_cost)
        sigma_x2_lat_cost= np.sqrt(var_x2_lat_cost)
        return cost_low_x2, u_x2_lat_cost, var_x2_lat_cost, sigma_x2_lat_cost

    def calculate_lower_cost_grad(x2, model, latent_cost_model_la, latent_cost_kernel):

        u_x2_cost, u_x2_lat_cost, var_x2_lat_cost = get_mean_and_var_cost(x2, latent_cost_model_la)
        sigma_x2_lat_cost = np.sqrt(var_x2_lat_cost)
        du_lat_cost_dx2, dvar_lat_cost_dx2, _= get_mean_variance_gradients_q1(latent_cost_kernel, Xt1, Yt1_latent_cost, x2, noise_cost)

        dsigma_lat_cost_dx2= (1/(2*sigma_x2_lat_cost))*dvar_lat_cost_dx2
        cost_low_x2 = get_lower_cost_bound(u_x2_lat_cost, var_x2_lat_cost)
        grad_cost_low= (du_lat_cost_dx2 - 1.96* dsigma_lat_cost_dx2)*cost_low_x2

        return grad_cost_low, du_lat_cost_dx2, dvar_lat_cost_dx2, dsigma_lat_cost_dx2

    x1= np.random.uniform(lower, upper, (1, D))

    Xt1= np.append(Xt, x1, axis= 0)
    u_x_cost_test, u_x_lat_cost_test, var_x_lat_cost_test = get_mean_and_var_cost(x1, latent_cost_model)
    cost_upper_x_test = get_upper_cost_bound(u_x_lat_cost_test, var_x_lat_cost_test)

    Yt1_latent_cost= np.append(Yt_latent_cost, np.log(cost_upper_x_test), axis=0)

    #update model accordingly
    latent_cost_model_la= gp.models.GPR((Xt1, Yt1_latent_cost), latent_cost_kernel)
    latent_cost_model_la.likelihood.variance.assign(noise_cost)

    x2_test = np.random.uniform(lower, upper, (1, D))
    x2_11= x2_test.copy(); x2_11[0,0]= x2_11[0,0]+ 0.0001;
    x2_12= x2_test.copy(); x2_12[0,0] = x2_12[0,0]- 0.0001
    x2_21= x2_test.copy(); x2_21[0,1] = x2_21[0,1]+ 0.0001
    x2_22= x2_test.copy(); x2_22[0,1] = x2_22[0,1]- 0.0001

    cost_low_x2_test, u_x2_test, var_x2_test, sigma_x2_test= calculate_lower_cost(x2_test, model, latent_cost_model_la)
    cost_low_x2_11, u_x2_11, var_x2_11, sigma_x2_11 = calculate_lower_cost(x2_11, model, latent_cost_model_la)
    cost_low_x2_12, u_x2_12, var_x2_12, sigma_x2_12 = calculate_lower_cost(x2_12, model, latent_cost_model_la)
    cost_low_x2_21, u_x2_21, var_x2_21, sigma_x2_21 = calculate_lower_cost(x2_21, model, latent_cost_model_la)
    cost_low_x2_22, u_x2_22, var_x2_22, sigma_x2_22 = calculate_lower_cost(x2_22, model, latent_cost_model_la)


    grad_app_cost_low_x2_dir1 = ((cost_low_x2_11 -cost_low_x2_test) / (x2_11[0, 0] - x2_test[0, 0]) + (cost_low_x2_test - cost_low_x2_12) / (x2_test[0, 0] - x2_12[0, 0])) / 2;
    grad_app_cost_low_x2_dir2 = ((cost_low_x2_21 - cost_low_x2_test) / (x2_21[0, 1] - x2_test[0, 1]) + (cost_low_x2_test - cost_low_x2_22) / (x2_test[0, 1] - x2_22[0, 1])) / 2;

    grad_app_cost_low_u_x2_dir1 = ((u_x2_11 -u_x2_test) / (x2_11[0, 0] - x2_test[0, 0]) + (u_x2_test - u_x2_12) / (x2_test[0, 0] - x2_12[0, 0])) / 2;
    grad_app_cost_low_u_x2_dir2 = ((u_x2_21 - u_x2_test) / (x2_21[0, 1] - x2_test[0, 1]) + (u_x2_test - u_x2_22) / (x2_test[0, 1] - x2_22[0, 1])) / 2;

    grad_app_var_x2_dir1 = ((var_x2_11 - var_x2_test) / (x2_11[0, 0] - x2_test[0, 0]) + (var_x2_test - var_x2_12) / (x2_test[0, 0] - x2_12[0, 0])) / 2;
    grad_app_var_x2_dir2 = ((var_x2_21 - var_x2_test) / (x2_21[0, 1] - x2_test[0, 1]) + (var_x2_test - var_x2_22) / (x2_test[0, 1] - x2_22[0, 1])) / 2;

    grad_app_sigma_x2_dir1 = ((sigma_x2_11 - sigma_x2_test) / (x2_11[0, 0] - x2_test[0, 0]) + (sigma_x2_test - sigma_x2_12) / (x2_test[0, 0] - x2_12[0, 0])) / 2;
    grad_app_sigma_x2_dir2 = ((sigma_x2_21 - sigma_x2_test) / (x2_21[0, 1] - x2_test[0, 1]) + (sigma_x2_test - sigma_x2_22) / (x2_test[0, 1] - x2_22[0, 1])) / 2;

    grad_cost_low_x2_test, du_lat_cost_dx2, dvar_lat_cost_dx2,  dsigma_lat_cost_dx2= calculate_lower_cost_grad(x2_test, model, latent_cost_model_la, latent_cost_kernel)

    grad_app_cost_low_x2_test = np.append(grad_app_cost_low_x2_dir1,grad_app_cost_low_x2_dir2, axis=0)
    grad_app_u_x2_test = np.append(grad_app_cost_low_u_x2_dir1,grad_app_cost_low_u_x2_dir2, axis=0)
    grad_app_var_x2_test = np.append(grad_app_var_x2_dir1 ,grad_app_var_x2_dir2 , axis=0)
    grad_app_sigma_x2_test = np.append(grad_app_sigma_x2_dir1 ,grad_app_sigma_x2_dir2 , axis=0)

    print('analytical_grad_cost_lower: \n{}, \napp_grad_cost_lower:\n{}'.format(grad_cost_low_x2_test, grad_app_cost_low_x2_test))
    print('analytical_grad_u_x2: \n{}, \napp_grad_u_x2:\n{}'.format(du_lat_cost_dx2,  grad_app_u_x2_test))
    print('analytical_grad_var_x2: \n{}, \napp_grad_var_x2:\n{}'.format(dvar_lat_cost_dx2,  grad_app_var_x2_test))
    print('analytical_grad_sigma_x2: \n{}, \napp_grad_sigma_x2:\n{}'.format(dsigma_lat_cost_dx2,  grad_app_sigma_x2_test))

import matplotlib.pyplot as plt
def temp_test_inner_opt():

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

    num_inner_opt_restarts= 10; num_outer_opt_restarts=0;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    x1= np.random.uniform(lower, upper, (1, D))

    Xt1= np.append(Xt, x1, axis= 0)
    u_x_cost_test, u_x_lat_cost_test, var_x_lat_cost_test = get_mean_and_var_cost(x1, latent_cost_model)
    cost_upper_x_test = get_upper_cost_bound(u_x_lat_cost_test, var_x_lat_cost_test)

    Yt1_latent_cost= np.append(Yt_latent_cost, np.log(cost_upper_x_test), axis=0)

    latent_cost_model_la= gp.models.GPR((Xt1, Yt1_latent_cost), latent_cost_kernel)
    latent_cost_model_la.likelihood.variance.assign(noise_cost)

    inner_opt= Ei_pu_la_optimize()
    x2_opt_value, x2_opt, ei_x2_opt, EI_X2_cepu_grid, X2_grid, x2_opt_sci,x2_opt_value_sci= inner_opt.maximize_ei_pu(kernel, latent_cost_model_la, Xt, noise, model, domain, num_inner_opt_restarts, f_best,
                       grid_opt_in, D, num_iter_max, noise_cost, latent_cost_kernel, Xt1, Yt1_latent_cost, Yt)

    plt.figure()
    plt.plot( X2_grid[:,0], EI_X2_cepu_grid[:,0], color= 'red')
    plt.scatter(X2_grid[:,0], EI_X2_cepu_grid[:,0], color= 'red')
    plt.scatter(x2_opt_sci[0, :], x2_opt_value_sci[:,0], color= 'blue')
    plt.show()

def temp_test_opt():

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

    num_inner_opt_restarts= 0; num_outer_opt_restarts=10;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    x1= np.random.uniform(lower, upper, (1, D))

    Xt1= np.append(Xt, x1, axis= 0)
    u_x_cost_test, u_x_lat_cost_test, var_x_lat_cost_test = get_mean_and_var_cost(x1, latent_cost_model)
    cost_upper_x_test = get_upper_cost_bound(u_x_lat_cost_test, var_x_lat_cost_test)

    Yt1_latent_cost= np.append(Yt_latent_cost, np.log(cost_upper_x_test), axis=0)

    latent_cost_model_la= gp.models.GPR((Xt1, Yt1_latent_cost), latent_cost_kernel)
    latent_cost_model_la.likelihood.variance.assign(noise_cost)

    opt= Ei_cepu_optimize()
    x_opt, x_opt_value, x_value_list_grid, X_grid, x_opt_sci, x_opt_value_sci= opt.maximize_ei_cepu(kernel, latent_cost_kernel, Xt, Yt, noise, model, latent_cost_model, domain, f_best,
                             num_inner_opt_restarts, num_outer_opt_restarts, grid_opt_in, grid_opt_out, D, Yt_latent_cost, num_iter_max, noise_cost)

    plt.figure()
    plt.plot(X_grid[:,0], x_value_list_grid[:,0], color= 'red', alpha= 0.5)
    plt.scatter(X_grid[:,0], x_value_list_grid[:,0], color= 'red', alpha= 0.5)
    plt.scatter(x_opt_sci[0, :], x_opt_value_sci[0,:])
    plt.show()

import matplotlib.pyplot as plt
def test_pucla():

    _, __, domain = sin_opt()
    D = 1; noise = 10 ** (-3)
    _, __, domain_cost= exp_cos_1d_opt()
    noise_cost = 10 ** (-3)

    lower = [domain[i][0] for i in range(len(domain))];
    upper = [domain[i][1] for i in range(len(domain))]

    Xt= np.random.uniform(lower, upper, (3,D))
    Yt= sin(Xt)
    Yt_cost= exp_cos_1d(Xt); log_Yt_cost= np.log(Yt_cost); Yt_latent_cost= log_Yt_cost.copy()

    disc = 101
    x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
    X_grid = x1_grid.reshape(-1, 1)
    Y_cost_grid=  exp_cos_1d(X_grid); log_Y_grid= np.log(Y_cost_grid)

    # plt.figure()
    # plt.plot(X_grid[:,0], Y_cost_grid[:,0], color='red')
    # plt.scatter(X_grid[:,0], Y_cost_grid[:,0], color='red')
    # plt.show()


    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(noise)

    latent_cost_kernel= gp.kernels.RBF()
    latent_cost_model= gp.models.GPR((Xt, log_Yt_cost), kernel)
    latent_cost_model.likelihood.variance.assign(noise_cost)

    f_best= np.min(Yt, axis=0); num_iter_max= 100;

    num_inner_opt_restarts= 0; num_outer_opt_restarts=10;
    grid_opt_in= True; grid_opt_out= True
    num_iter_max = 100;

    x_test= Xt[0,:].reshape(1,-1)
    xpl= x_test+0.2
    xng= x_test-0.2

    '''calculate per unit cost lookahead values'''
    opt= Ei_cepu_optimize()
    cepu_neg_x_test, lower_cost_term_test, upper_cost_term_test, one_step_opt_term_test= \
                        opt.ei_cepu_negative(x_test, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)

    ucostX_lat_grid_test, varcostX_lat_grid_test, sigmacostX_lat_grid_test = get_mean_var_std(X_grid, opt.latent_cost_model_la)
    ucost_grid_test= np.exp(ucostX_lat_grid_test); u_up_grid_test= np.exp(ucostX_lat_grid_test+1.96*sigmacostX_lat_grid_test);
    u_low_grid_test= np.exp(ucostX_lat_grid_test-1.96*sigmacostX_lat_grid_test)

    cepu_neg_pl, lower_cost_term_pl, upper_cost_term_pl, one_step_opt_term_pl= \
                        opt.ei_cepu_negative(xpl, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)

    ucostX_lat_grid_pl, varcostX_lat_grid_pl, sigmacostX_lat_grid_pl = get_mean_var_std(X_grid, opt.latent_cost_model_la)
    ucost_grid_pl= np.exp(ucostX_lat_grid_pl); u_up_grid_pl= np.exp(ucostX_lat_grid_pl+1.96*sigmacostX_lat_grid_pl);
    u_low_grid_pl= np.exp(ucostX_lat_grid_pl-1.96*sigmacostX_lat_grid_pl)

    cepu_neg_ng, lower_cost_term_ng, upper_cost_term_ng, one_step_opt_term_ng= \
                            opt.ei_cepu_negative(xng, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)

    ucostX_lat_grid_ng, varcostX_lat_grid_ng, sigmacostX_lat_grid_ng = get_mean_var_std(X_grid, opt.latent_cost_model_la)
    ucost_grid_ng= np.exp(ucostX_lat_grid_ng); u_up_grid_ng= np.exp(ucostX_lat_grid_ng+1.96*sigmacostX_lat_grid_ng);
    u_low_grid_ng= np.exp(ucostX_lat_grid_ng-1.96*sigmacostX_lat_grid_ng)

    print('ei_pu_lower:{}, ei_pu_upper:{}, one_step_opt_ei_pu:{} calculated at previously evaluated point'.
                                    format(lower_cost_term_test, upper_cost_term_test, one_step_opt_term_test))
    print('ei_pu_lower_pl:{}, ei_pu_upper_pl:{}, one_step_opt_ei_pu_pl:{} calculated at previously evaluated point'.
                                    format(lower_cost_term_pl, upper_cost_term_pl, one_step_opt_term_pl))
    print('ei_pu_lower_ng:{}, ei_pu_upper_ng:{}, one_step_opt_ei_pu_ng:{} calculated at previously evaluated point'.
                                    format(lower_cost_term_ng, upper_cost_term_ng, one_step_opt_term_ng))

    cepu_x_test= np.atleast_2d(-cepu_neg_x_test)
    cepu_pl= np.atleast_2d(-cepu_neg_pl)
    cepu_ng= np.atleast_2d(-cepu_neg_ng)

    cepu_grid= np.empty([disc,1])

    for i in range(disc):
        xi= X_grid[i, :].reshape(1,-1)
        cepu_grid[i, :], _, _, _= opt.ei_cepu_negative(xi, model, latent_cost_model, f_best, Yt_latent_cost, kernel, latent_cost_kernel,
                         noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max, num_inner_opt_restarts, Yt, D)
        cepu_grid[i,:]*= -1

    evaluated_grid= np.empty([Xt.shape[0], 1])
    for i in range(Xt.shape[0]):
        xi = Xt[i, :].reshape(1, -1)

        evaluated_grid[i, :], _, _, _ = opt.ei_cepu_negative(xi, model, latent_cost_model, f_best, Yt_latent_cost,
                                                        kernel, latent_cost_kernel,
                                                        noise_cost, Xt, noise, domain, grid_opt_in, num_iter_max,
                                                        num_inner_opt_restarts, Yt, D)
        evaluated_grid[i, :] *= -1

    '''calculate ei values'''
    uX_grid, varX_grid, sigmaX_grid= get_mean_var_std(X_grid, model)

    ux_test, varx_test, sigmax_test= get_mean_var_std(x_test, model)
    ux_pl, varx_pl, sigmax_pl= get_mean_var_std(xpl, model)
    ux_ng, varx_ng, sigmax_ng= get_mean_var_std(xng, model)

    ei_test= ei(sigmax_test, ux_test, f_best)
    ei_pl= ei(sigmax_pl, ux_pl, f_best)
    ei_ng= ei(sigmax_ng, ux_ng, f_best)

    EI_grid= ei(sigmaX_grid, uX_grid, f_best)

    '''calculate cost values'''
    ucostX_lat_grid, varcostX_lat_grid, sigmacostX_lat_grid = get_mean_var_std(X_grid, latent_cost_model)
    ucost_grid= np.exp(ucostX_lat_grid); u_up_grid= np.exp(ucostX_lat_grid+1.96*sigmacostX_lat_grid);
    u_low_grid= np.exp(ucostX_lat_grid-1.96*sigmacostX_lat_grid)

    ucostx_lat_test, varcostx_lat_test, sigmacostx_lat_test= get_mean_var_std(x_test, latent_cost_model)
    ucost_test = np.exp(ucostx_lat_test)
    ucostx_lat_pl, varcostx_lat_pl, sigmacostx_lat_pl = get_mean_var_std(xpl, latent_cost_model)
    ucost_pl = np.exp(ucostx_lat_pl);
    ucostx_lat_ng, varcostx_lat_ng, sigmacostx_lat_ng = get_mean_var_std(xng, latent_cost_model)
    ucost_ng = np.exp(ucostx_lat_ng);

    '''lookahead ei_pu'''
    la_ng= EI_grid/u_low_grid_ng
    la_pl= EI_grid/u_low_grid_pl
    la_test= EI_grid/u_low_grid_test

    plt.figure()
    plt.title('pucla')
    plt.plot(X_grid[:,0], cepu_grid[:,0], color= 'red')
    plt.scatter(X_grid[:,0], cepu_grid[:,0], color= 'red')

    plt.scatter(x_test[:,0], cepu_x_test[:,0], color= 'green')
    plt.scatter(Xt[:,0], evaluated_grid[:,0], color= 'green')
    plt.scatter(xpl[:,0], cepu_pl[:,0], color= 'blue')
    plt.scatter(xng[:,0], cepu_ng[:,0], color= 'blue')
    plt.show()

    plt.figure()
    plt.title('EI')
    plt.scatter(X_grid[:,0], EI_grid[:,0], color= 'red')
    plt.plot(X_grid[:,0], EI_grid[:,0], color= 'red')
    plt.scatter(x_test[0,:], ei_test[0,:], color= 'green')
    plt.scatter(xpl[0,:], ei_pl[0,:], color= 'blue')
    plt.scatter(xng[0,:], ei_ng[0,:], color= 'blue')
    plt.show()

    plt.figure()
    plt.title('cost')
    plt.scatter(X_grid[:,0], ucost_grid[:,0], color= 'red')
    plt.scatter(X_grid[:,0], u_up_grid[:,0], color= 'grey')
    plt.scatter(X_grid[:,0], u_low_grid[:,0], color= 'grey')

    plt.scatter(x_test[0, :], ucost_test[0,:], color= 'green')
    plt.scatter(xpl[0, :], ucost_pl[0,:], color= 'blue')
    plt.scatter(xng[0, :], ucost_ng[0,:], color= 'blue')

    plt.show()

    plt.figure()
    plt.title('lookahead pu')
    plt.plot(X_grid[:, 0], la_ng[:, 0], color='blue')
    plt.plot(X_grid[:, 0], la_pl[:, 0], color='orange')
    plt.plot(X_grid[:, 0], la_test[:, 0], color='green')

    plt.scatter(xng, np.zeros([1, 1]), color= 'blue')
    plt.scatter(xpl, np.zeros([1, 1]), color= 'orange')
    plt.scatter(x_test, np.zeros([1, 1]), color= 'green')

    plt.show()