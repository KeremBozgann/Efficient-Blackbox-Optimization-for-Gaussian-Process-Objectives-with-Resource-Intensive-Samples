import numpy as np
from scipy.stats import norm


def uniformly_choose_from_domain(domain, num_samples):
    X= np.empty([num_samples, len(domain)])
    # Y= np.empty([num_samples, 1])

    for i in range(len(domain)):
        xi= np.random.uniform(domain[i][0], domain[i][1], (num_samples,1))
        X[:, i]= xi[:,0]
    return X

def initial_training_synthetic(domain ,num_samples, objective_func, cost_func):

    X= uniformly_choose_from_domain(domain, num_samples)
    Y= objective_func(X)
    Y_cost= cost_func(X)

    return X, Y, Y_cost

def get_grid(domain, disc):

    x_list = []
    for i in range(len(domain)):
        xi = np.linspace(domain[i][0], domain[i][1], disc)
        x_list.append(xi)
    X_temp = np.meshgrid(*x_list)

    X_grid = np.empty([disc ** len(domain), len(domain)])

    for i in range(len(domain)):
        X_grid[:, i] = X_temp[i].flatten()
    del X_temp

    return X_grid


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

def ei(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))

    return EI_x

def dcumulative_du(f_best, sigma_x, u_x):

    return -1/sigma_x*norm.pdf((f_best- u_x)/sigma_x)


def dcumulative_dsigma(f_best, sigma_x, u_x):

    return -(f_best-u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)


def dnormal_du(f_best, u_x, sigma_x):

    return (f_best- u_x)/(sigma_x**2)*norm.pdf((f_best-u_x)/sigma_x)


def dnormal_dsigma(f_best, u_x, sigma_x):

    return ((f_best-u_x)**2)/(sigma_x**3)*norm.pdf((f_best- u_x)/sigma_x)


def dcumulative_dfbest(f_best, sigma_x, u_x):

    return 1/sigma_x*norm.pdf((f_best- u_x)/sigma_x)

def dnormal_dfbest(f_best, u_x, sigma_x):

    return -(f_best - u_x) / (sigma_x** 2) * norm.pdf((f_best - u_x) / sigma_x)

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

def get_posterior_covariance(x, x1, Xt, kernel, noise):

    K_inv= np.linalg.inv((kernel.K(Xt, Xt)).numpy()+ np.eye(Xt.shape[0])*noise)
    temp= np.matmul((kernel.K(x,Xt)).numpy(), K_inv)
    result= (kernel.K(x,x1)).numpy()- np.matmul(temp, ((kernel.K(x1, Xt)).numpy()).T)

    return result


def get_one_step_mean_and_variance(u0_x2, var0_x2, Z, sigma0):

    u1_x2= u0_x2 + np.matmul(sigma0,Z)

    var1_x2= var0_x2- np.matmul(sigma0, sigma0.T)


    return u1_x2, var1_x2


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

def get_dsigma0_dx1_q1(x2_opt, x1, Xt, kernel, noise, dK0_x1_dx1, D, L0_x1, L0_x1_inv, L0_x1_inv_T):

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

    du1_x2_dx1= np.matmul(dsigma0_dx1, np.atleast_2d(Z))

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


def dEI1_x2_dx1(f_best, f_best1,  kernel, Xt, Yt, noise, x2_opt, x1, D, Z, u0_x2, var0_x2, L0_x1, L0_x1_inv, L0_x1_inv_T, y1):

    Z= np.atleast_2d(Z)
    du0_x1_dx1, dK0_x1_dx1, K0_x1 =  get_mean_variance_gradients_q1(kernel, Xt, Yt, x1, noise)
    dsigma0_dx1, sigma0, dL0_x1_dx1= get_dsigma0_dx1_q1(x2_opt, x1, Xt, kernel, noise, dK0_x1_dx1, D,
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

def create_grid(disc, domain):

    x_list = []
    for i in range(len(domain)):
        xi = np.linspace(domain[i][0], domain[i][1], disc)
        x_list.append(xi)
    X_temp = np.meshgrid(*x_list)

    X_grid = np.empty([disc ** len(domain), len(domain)])

    for i in range(len(domain)):
        X_grid[:, i] = X_temp[i].flatten()

    return X_grid

def uniform_sampling(num_samples_uni, domain):
    lower = [];
    upper = []
    for i in range(len(domain)):
        lower.append(domain[i][0])
        upper.append(domain[i][1])

    D= len(domain)
    X= np.random.uniform(lower, upper, (num_samples_uni, D))

    return X

def ei_sampling(num_samples_uni, num_samples_ei, domain, model, f_best):

    X_uni= uniform_sampling(num_samples_uni, domain)
    u_X, var_X, sigma_X= get_mean_var_std(X_uni, model)
    ei_X= ei(sigma_X, u_X, f_best)
    choice= np.random.choice(np.arange(X_uni.shape[0]), num_samples_ei, p= ei_X[:, 0]/np.sum(ei_X, axis=0))

    X_ei= X_uni[choice, :]
    values_ei= ei_X[choice, :]
    return X_ei, values_ei

from pyDOE import *

def scale(domain, X):

    for i in range(len(domain)):
        lowi= domain[i][0]
        highi= domain[i][1]
        middle=( highi+lowi)/2
        X[:,i]= (X[:,i]-0.5)*(highi- lowi)+middle

    return X

def ei_lthc_sampling(num_lthc_samples, num_ei_samples, domain, model, f_best):

    D= len(domain)
    X_lthc = lhs(D, samples=num_lthc_samples, criterion='maximin')
    X_lthc= scale(domain, X_lthc)

    u_X, var_X, sigma_X= get_mean_var_std(X_lthc, model)
    ei_X= ei(sigma_X, u_X, f_best)
    choice= np.random.choice(np.arange(X_lthc.shape[0]), num_ei_samples, p= ei_X[:, 0]/np.sum(ei_X, axis=0))

    X_ei= X_lthc[choice, :]
    values_ei= ei_X[choice, :]


    return X_ei, values_ei