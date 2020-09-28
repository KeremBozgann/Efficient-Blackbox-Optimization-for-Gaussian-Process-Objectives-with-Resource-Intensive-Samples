import sys
sys.path.append('./functions')
from branin import branin
from sine import sin
import numpy as np
import gpflow as gp

from gp_gradients import dEI_dx
from gp_gradients import  mean_variance_gradients
from gp_gradients import dcumulative_du, dcumulative_dsigma
from gp_gradients import dEI1_x2_dx1
from gp_gradients import two_opt_EI_optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('./bo_cost_budget_cost_domain')
from Acquisitions import EI

from scipy.stats import norm

from scipy import spatial

from gp_gradients import EI1_x2_optimize

def gradient_using_closest_point_in_the_grid_1d(x, X, model, f_best):

    x_nearest = X[spatial.KDTree(X).query(x)[1]]

    u_x, var_x= model.predict_f(x); sigma_x= np.sqrt(var_x)
    u_x_n, var_x_n = model.predict_f(x_nearest); sigma_x_n= np.sqrt(var_x_n)
    ei_x= EI(sigma_x, u_x, f_best); ei_x_n= EI(sigma_x_n, u_x_n, f_best)

    dx= x- x_nearest; dy= ei_x- ei_x_n
    dy_dx= dy/dx

    return dy_dx, x_nearest, ei_x_n

def test_on_branin():
    '''test on branin function'''

    domain= [[-5,10], [0,15]]
    noise= 10**(-4)

    X1= np.random.uniform(domain[0][0], domain[0][1], size= (10, 1))
    X2= np.random.uniform(domain[1][0], domain[1][1], size= (10, 1))

    Xt= np.append(X1,X2, axis=1)
    Yt= branin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel); model.likelihood.variance.assign(noise)


    '''grid'''
    disc = 20
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    x2 = np.linspace(domain[1][0], domain[1][1], disc)
    x1_max, x2_max, x1_min, x2_min = np.max(x1), np.max(x2), np.min(x1), np.min(x2)
    X1, X2 = np.meshgrid(x1, x2);

    X1_flat, X2_flat = X1.flatten(), X2.flatten();
    X1_flat, X2_flat = X1.reshape(-1, 1), X2.reshape(-1, 1)
    X = np.append(X1_flat, X2_flat, axis=1)

    Y = branin(X)

    '''point to evaluate gradient'''
    x1= np.random.uniform(domain[0][0], domain[0][1], size= (1,1));
    x2= np.random.uniform(domain[1][0], domain[1][1], size= (1,1));

    x= np.append(x1,x2, axis=1)
    x11= x.copy(); x11[0,0]= x11[0,0]+0.0001;
    x12= x.copy(); x12[0,0] = x12[0,0]- 0.0001
    x21= x.copy(); x21[0,1] = x21[0,1]+ 0.0001
    x22= x.copy(); x22[0,1] = x22[0,1]- 0.0001

    print('x11:{}, \nx12:{}, \nx21:{} \nx22:{}'.format(x11, x12, x21, x22))

    f_best= np.min(Yt, axis=0)
    u_x, var_x= model.predict_f(x); sigma_x= np.sqrt(var_x)
    u_x11, var_x11= model.predict_f(x11); sigma_x11= np.sqrt(var_x11)
    u_x12, var_x12= model.predict_f(x12); sigma_x12= np.sqrt(var_x12)
    u_x21, var_x21= model.predict_f(x21); sigma_x21= np.sqrt(var_x21)
    u_x22, var_x22= model.predict_f(x22); sigma_x22= np.sqrt(var_x22)



    '''predictions of model to plot acquisition on grid'''
    u_X, var_X= model.predict_f(X); sigma_X= np.sqrt(var_X)
    EI_grid= EI(sigma_X, u_X, f_best)


    EI_x= EI(sigma_x, u_x, f_best)
    EI_x11= EI(sigma_x11, u_x11, f_best)
    EI_x12= EI(sigma_x12, u_x12, f_best)
    EI_x21= EI(sigma_x21, u_x21, f_best)
    EI_x22= EI(sigma_x22, u_x22, f_best)

    EI_gradient= dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise)

    grad_app_dir1 = ((EI_x11-EI_x)/(x11[0,0]-x[0,0])+(EI_x-EI_x12)/(x[0,0]- x12[0,0]))/2;
    grad_app_dir2 = ((EI_x21-EI_x)/(x21[0,1]-x[0,1])+(EI_x-EI_x22)/(x[0,1]- x22[0,1]))/2;

    grad_app= np.append(grad_app_dir1, grad_app_dir2, axis=0)

    print('EI_gradient: \n{}, \ngrad_app:\n{}'.format(EI_gradient, grad_app))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('ei acquisition')

    ax.scatter3D(X[:,0], X[:,1], EI_grid[:,0], color= 'red', label= 'acquisition on grid')

    ax.scatter3D(x[0,0], x[0,1], EI_x[0,0], color= 'blue', label='gradient point')

    plt.legend()
    plt.show()



def test_on_1d_sin():
    '''test on 1d sin function'''

    noise= 10**(-4)

    domain =[[-2,2]]

    Xt = np.random.uniform(domain[0][0], domain[0][1], size=(3, 1))

    Yt = sin(Xt)

    kernel = gp.kernels.RBF()
    model = gp.models.GPR((Xt, Yt), kernel);
    model.likelihood.variance.assign(noise)


    '''grid'''

    disc = 100
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    X = x1.reshape(-1, 1)
    Y = sin(X)


    '''point to evaluate gradient'''
    x= np.random.uniform(domain[0][0], domain[0][1], size= (1,1));
    x1= x+ 0.0001; x2= x- 0.0001


    f_best= np.min(Yt, axis=0)
    u_x, var_x= model.predict_f(x); sigma_x= np.sqrt(var_x)
    u_x1, var_x1= model.predict_f(x1); sigma_x1= np.sqrt(var_x1)
    u_x2, var_x2= model.predict_f(x2); sigma_x2= np.sqrt(var_x2)


    '''predictions of model to plot acquisition on grid'''
    u_X, var_X= model.predict_f(X); sigma_X= np.sqrt(var_X)

    EI_grid= EI(sigma_X, u_X, f_best)

    '''ei and its gradient at point '''
    EI_x= EI(sigma_x, u_x, f_best)
    EI_x1= EI(sigma_x1, u_x1, f_best)
    EI_x2= EI(sigma_x2, u_x2, f_best)
    grad_x= dEI_dx(f_best, u_x, sigma_x, kernel, Xt, Yt, x, noise)

    grad_app= ((EI_x1-EI_x)/(x1-x)+ (EI_x- EI_x2)/(x- x2))/2
    grad_n, x_nearest, EI_x_nearest = gradient_using_closest_point_in_the_grid_1d(x, X, model, f_best)


    fig = plt.figure()
    plt.title('EI acquisition')

    plt.scatter(X[:,0], EI_grid[:,0], color= 'red', label= 'acquisition on grid', alpha= 0.5)

    plt.scatter(x[0,0], EI_x[0,0], color= 'blue', label='gradient point')

    plt.scatter(x_nearest[0,0], EI_x_nearest[0,0], color= 'green', label='nearest point')

    plt.legend()
    plt.show()



    print('x:{}, x_nearest:{}'.format(x, x_nearest))
    # print('grad_x:{}, grad_app:{}'.format(grad_x, grad_n))
    print('grad_x:{}, grad_app:{}'.format(grad_x, grad_app))

    return grad_x, grad_app, grad_n, x, x_nearest

def test_mean_variance_gradients_1d(model, kernel, x, X, Xt, Yt, noise ):

    domain= [[-2, 2]]

    Xt = np.random.uniform(domain[0][0], domain[0][1], size=(3, 1))

    Yt = sin(Xt)

    kernel = gp.kernels.RBF()
    model = gp.models.GPR((Xt, Yt), kernel);
    model.likelihood.variance.assign(10 ** (-4))

    '''point to evaluate gradient'''
    x= np.random.uniform(domain[0][0], domain[0][1], size= (1,1));
    x1= x+0.01; x2= x- 0.01
    u_x, var_x= model.predict_f(x); sigma_x= np.sqrt(var_x)
    u_x1, var_x1= model.predict_f(x1); sigma_x1= np.sqrt(var_x1)
    u_x2, var_x2= model.predict_f(x2); sigma_x2= np.sqrt(var_x2)
    u_grad_app= ((u_x1-u_x)/(x1-x)+ (u_x- u_x2)/((x-x2)))/2
    var_grad_app= ((var_x1-var_x)/(x1-x)+ (var_x- var_x2)/((x-x2)))/2
    du_dx, dvar_dx= mean_variance_gradients(kernel, Xt, Yt, x, noise)

    print('analytical mean gradient:{}, approximated mean gradient:{}'.format(du_dx, u_grad_app) )
    print('analytical variance gradient:{}, approximated variance gradient:{}'.format(dvar_dx, var_grad_app) )

    u_X, var_X= model.predict_f(X); sigma_X= np.sqrt(var_X)

    plt.figure()
    plt.title('mean gradient check')
    plt.scatter(X, u_X, color= 'red', label= 'mean on grid')
    plt.scatter(x, u_x, color= 'blue', label= 'grad test point1')

    plt.legend()
    plt.show()

    plt.figure()
    plt.title('var gradient chcek')
    plt.scatter(X, var_X, color='red', label= 'sigma on grid')
    plt.scatter(x, var_x, color='blue', label='grad test point1')
    plt.legend()
    plt.show()


def test_cumulative_derivative_sin_1d():


    noise= 10**(-4)

    domain= [[-2, 2]]

    Xt = np.random.uniform(domain[0][0], domain[0][1], size=(3, 1))

    Yt = sin(Xt)

    f_best= np.amin(Yt, axis=0)

    kernel = gp.kernels.RBF()
    model = gp.models.GPR((Xt, Yt), kernel);
    model.likelihood.variance.assign(noise)

    '''point to evaluate gradient'''
    # x= np.random.uniform(domain[0][0], domain[0][1], size= (1,1));
    x= np.array([[-1.2]])
    x1= x+0.01; x2= x- 0.01
    u_x, var_x= model.predict_f(x); sigma_x= np.sqrt(var_x)
    u_x1, var_x1= model.predict_f(x1); sigma_x1= np.sqrt(var_x1)
    u_x2, var_x2= model.predict_f(x2); sigma_x2= np.sqrt(var_x2)

    gama_x = (f_best - u_x) / sigma_x; gama_x1= (f_best- u_x1)/sigma_x1;
    gama_x2= (f_best- u_x2)/sigma_x2

    cum_x = norm.cdf(gama_x); cum_x1= norm.cdf(gama_x1); cum_x2= norm.cdf(gama_x2)

    cum_deriv_app= ((cum_x1-cum_x)/(x1-x)+ (cum_x- cum_x2)/(x- x2))/2

    du_dx, dvar_dx= mean_variance_gradients(kernel, Xt, Yt, x, noise)
    dsigma_dx= (1/(2*sigma_x))*dvar_dx

    mean_deriv_app = ((u_x1-u_x)/(x1-x)+ (u_x- u_x2)/(x- x2))/2
    var_deriv_app=  ((var_x1-var_x)/(x1-x)+ (var_x- var_x2)/(x- x2))/2
    sigma_deriv_app = ((sigma_x1 - sigma_x) / (x1 - x) + (sigma_x - sigma_x2) / (x - x2)) / 2

    dcum_du = dcumulative_du(f_best, sigma_x, u_x)
    cumulative_app_u_deriv= ((cum_x1-cum_x)/(u_x1-u_x)+ (cum_x- cum_x2)/(u_x- u_x2))/2

    dcum_dsigma= dcumulative_dsigma(f_best, sigma_x, u_x)
    cumulative_app_sigma_deriv=  ((cum_x1-cum_x)/(sigma_x1-sigma_x)+ (cum_x- cum_x2)/(sigma_x- sigma_x2))/2


    dcumulative_dx = dcumulative_du(f_best, sigma_x, u_x) * du_dx + \
                     dcumulative_dsigma(f_best, sigma_x, u_x) * dsigma_dx


    print('analytical mean gradient:{}, approximated mean gradient:{}'.format(dcumulative_dx, cum_deriv_app) )


    disc = 100
    x1 = np.linspace(domain[0][0], domain[0][1], disc)
    X = x1.reshape(-1, 1)
    Y = sin(X)


    u_X, var_X= model.predict_f(X); sigma_X= np.sqrt(var_X)
    gama_X = (f_best - u_X) / sigma_X
    cum_X = norm.cdf(gama_X)

    plt.figure()
    plt.title('mean gradient check')
    plt.scatter(X, cum_X, color= 'red', label= 'cumulative on grid')
    plt.scatter(x, cum_x, color= 'blue', label= 'grad test point1')

    plt.legend()
    plt.show()


def test_EI1_x2_optimize_sin_1d():
    domain= [[-2,2]]
    noise= (10**(-4))
    d=1
    num_inner_restarts = 10*d
    Xt= np.random.uniform(domain[0][0], domain[0][1], (3,1))
    Yt= sin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel); model.likelihood.variance.assign(noise)

    x1= np.random.uniform(domain[0][0], domain[0][1], (1,1))
    Z= np.random.normal()

    f_best= np.min(Yt,axis=0)

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    sigma0_x1= np.sqrt(var0_x1)


    '''inner loop optimization with random restarts'''
    x2_opt_list=[]; opt_value_list= []; grad_list= []
    for i in range(num_inner_restarts):
        x20 = np.random.uniform(domain[0][0], domain[0][1], (1, 1))
        x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, f_best, domain)
        x2_cand_grad = - x2_cand_grad
        x2_cand = x2_cand.reshape(1, -1)
        print(result)
        x2_opt_list.append(x2_cand); opt_value_list.append(cand_value); grad_list.append(x2_cand_grad)

    index_opt= int(np.argmax(np.array(opt_value_list)))
    x2_opt= x2_opt_list[index_opt]
    opt_value= opt_value_list[index_opt]
    x2_opt_grad= grad_list[index_opt]



    '''compare with grid'''
    X_grid = np.linspace(domain[0][0], domain[0][1], 101);
    X_grid = X_grid.reshape(-1, 1)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)

    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    model1= gp.models.GPR((Xt1, Yt1), kernel)
    model1.likelihood.variance.assign(noise)

    u1_X_grid, var1_X_grid= model1.predict_f(X_grid); u1_X_grid= u1_X_grid.numpy()
    var1_X_grid= var1_X_grid.numpy(); sigma1_X_grid= np.sqrt(var1_X_grid)

    f_best1= np.min(Yt1, axis=0)
    EI1_X_grid= EI(sigma1_X_grid, u1_X_grid, f_best1)

    '''calculate approximate derivative'''
    x2_opt1= x2_opt+0.0001; x2_opt2= x2_opt- 0.0001
    u_x2_opt1, var_x2_opt1= model1.predict_f(x2_opt1); u_x2_opt1= u_x2_opt1.numpy(); var_x2_opt1= var_x2_opt1.numpy()
    u_x2_opt, var_x2_opt= model1.predict_f(x2_opt); u_x2_opt= u_x2_opt.numpy(); var_x2_opt= var_x2_opt.numpy()
    u_x2_opt2, var_x2_opt2= model1.predict_f(x2_opt2); u_x2_opt2= u_x2_opt2.numpy(); var_x2_opt2= var_x2_opt2.numpy()
    sigma_x2_opt1= np.sqrt(var_x2_opt1); sigma_x2_opt2= np.sqrt(var_x2_opt2); sigma_x2_opt= np.sqrt(var_x2_opt)
    ei_x2_opt1= EI(sigma_x2_opt1, u_x2_opt1, f_best1); ei_x2_opt2= EI(sigma_x2_opt2, u_x2_opt2, f_best1)
    ei_x2_opt= EI(sigma_x2_opt, u_x2_opt, f_best1)
    app_deriv= ((ei_x2_opt1- ei_x2_opt)/(x2_opt1-x2_opt)+(ei_x2_opt- ei_x2_opt2)/(x2_opt- x2_opt2))/2
    print('app_deriv:{}, x2_opt_deriv:{}'.format(app_deriv, x2_opt_grad))

    '''comparison plots'''
    plt.figure()
    plt.plot(X_grid, EI1_X_grid, color= 'red', alpha= 0.5)
    plt.scatter(np.array(x2_opt_list), np.array(opt_value_list), color= 'blue')
    plt.scatter(x2_opt, opt_value, color= 'green')
    plt.scatter(x20, np.zeros([1,1]), color= 'black', marker= 'X')
    plt.show()

from branin import branin

def test_EI1_x2_optimize_branin_2d():
    domain= [[-5, 10], [0, 15]]
    noise= (10**(-4))
    d=2
    num_inner_restarts = 10*d
    Xt= np.random.uniform([domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], (3,2))
    Yt= branin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel); model.likelihood.variance.assign(noise)

    x1= np.random.uniform([domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], (1,2))
    Z= np.random.normal()

    f_best= np.min(Yt,axis=0)

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    sigma0_x1= np.sqrt(var0_x1)


    '''inner loop optimization with random restarts'''
    x2_opt_list=[]; opt_value_list= []; grad_list= []; x20_list= []
    for i in range(num_inner_restarts):
        x20 = np.random.uniform([domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], (1,2))
        x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, f_best, domain)
        x2_cand = x2_cand.reshape(1, -1)

        print(result)
        x2_opt_list.append(x2_cand); opt_value_list.append(cand_value); grad_list.append(x2_cand_grad)
        x20_list.append(x20)

    index_opt= int(np.argmax(np.array(opt_value_list)))
    x2_opt= x2_opt_list[index_opt]
    opt_value= opt_value_list[index_opt]
    print('opt_value', opt_value)
    x2_opt_grad= grad_list[index_opt]
    x20_opt= x20_list[index_opt]
    '''compare with grid'''
    x_grid1 = np.linspace(domain[0][0], domain[0][1], 21);
    x_grid2=  np.linspace(domain[1][0], domain[1][1], 21);
    X1, X2= np.meshgrid(x_grid1, x_grid2); X1 = (X1.flatten()).reshape(-1,1); X2= (X2.flatten()).reshape(-1,1)
    X_grid = np.append(X1, X2, axis=1)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    L0_x1=  np.linalg.cholesky(var0_x1)
    y1= u0_x1+ np.matmul(L0_x1,Z_conv)

    Xt1= np.append(Xt, x1, axis=0)
    Yt1= np.append(Yt, y1, axis=0)

    model1= gp.models.GPR((Xt1, Yt1), kernel)
    model1.likelihood.variance.assign(noise)

    u1_X_grid, var1_X_grid= model1.predict_f(X_grid); u1_X_grid= u1_X_grid.numpy()
    var1_X_grid= var1_X_grid.numpy(); sigma1_X_grid= np.sqrt(var1_X_grid)

    f_best1= np.min(Yt1, axis=0)
    EI1_X_grid= EI(sigma1_X_grid, u1_X_grid, f_best1)

    '''calculate approximate derivative'''
    x11= x2_opt.copy(); x11[0,0]= x11[0,0]+0.0001; x12= x2_opt.copy(); x12[0,0]-= 0.0001
    x21= x2_opt.copy(); x21[0,1]= x21[0,1]+0.0001; x22= x2_opt.copy(); x22[0,1]-= 0.0001

    u_x11, var_x11= model1.predict_f(x11); u_x11= u_x11.numpy(); var_x11= var_x11.numpy()
    u_x12, var_x12= model1.predict_f(x12); u_x12= u_x12.numpy(); var_x12= var_x12.numpy()
    u_x21, var_x21= model1.predict_f(x21); u_x21= u_x21.numpy(); var_x21= var_x21.numpy()
    u_x22, var_x22= model1.predict_f(x22); u_x22= u_x22.numpy(); var_x22= var_x22.numpy()
    u_x2_opt, var_x2_opt= model1.predict_f(x2_opt); u_x2_opt= u_x2_opt.numpy(); var_x2_opt= var_x2_opt.numpy()

    sigma_x11= np.sqrt(var_x11); sigma_x12= np.sqrt(var_x12); sigma_x21= np.sqrt(var_x21);
    sigma_x22= np.sqrt(var_x22);  sigma_x2_opt= np.sqrt(var_x2_opt)

    ei_x11= EI(sigma_x11, u_x11, f_best1); ei_x12= EI(sigma_x12, u_x12, f_best1);
    ei_x21= EI(sigma_x21, u_x21, f_best1); ei_x22= EI(sigma_x22, u_x22, f_best1);
    ei_x2_opt= EI(sigma_x2_opt, u_x2_opt, f_best1)

    app_deriv1= ((ei_x11[0,0]- ei_x2_opt[0,0])/(x11[0,0]-x2_opt[0,0])+\
                                    (ei_x2_opt[0,0]- ei_x12[0,0])/(x2_opt[0,0]- x12[0,0]))/2
    app_deriv2= ((ei_x21[0,0]- ei_x2_opt[0,0])/(x21[0,1]-x2_opt[0,1])+
                                    (ei_x2_opt[0,0]- ei_x22[0,0])/(x2_opt[0,1]- x22[0,1]))/2
    app_deriv= np.array([app_deriv1, app_deriv2])
    print('app_deriv:\n{}\nx2_opt_deriv:\n{}'.format(app_deriv, x2_opt_grad))

    '''comparison plots'''

    x2_list = np.empty([0, 2])
    for i in range(len(x2_opt_list)):
        x2_list = np.append(x2_list, x2_opt_list[i], axis=0)

    value_list= np.asarray(opt_value_list)
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(X_grid[:,0],X_grid[:,1], EI1_X_grid[:,0], color= 'red', alpha= 0.5, label= 'ei1_x grid')
    ax.scatter(x2_list[:,0], x2_list[:,1], value_list[:,0], color= 'green', label= 'found optimums at restarts', alpha =0.5)
    ax.scatter(x2_opt[0,0], x2_opt[0,1], opt_value[0], color='blue', label= 'best optimum')
    # ax.scatter(x20_opt[0,0], x20_opt[0,1], np.zeros(1,), color='black', marker='X', label= 'initial x2- best optimum')

    plt.show()
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('y');


def test_two_step_optimization_1d_sin():

    domain= [[-2, 2]]
    noise= (10**(-4))
    D=1
    num_inner_opt_restarts = 10#10*D
    num_outer_opt_restarts= 10#10*D

    monte_carlo_samples= 10

    Xt= np.random.uniform(domain[0][0], domain[0][1], (3,1))
    Yt= sin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel); model.likelihood.variance.assign(noise)

    x1= np.random.uniform(domain[0][0], domain[0][1], (1,1))
    Z= np.random.normal(0.0, 1.0, [1,1])

    f_best= np.min(Yt,axis=0)

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    sigma0_x1= np.sqrt(var0_x1)
    #
    #
    # '''inner loop optimization with random restarts'''
    # x2_opt_list=[]; opt_value_list= []; grad_list= []; x20_list= []
    # for i in range(num_inner_restarts):
    #     x20 = np.random.uniform([domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], (1,2))
    #     x2_cand, cand_value, x2_cand_grad, result = EI1_x2_optimize(x20, u0_x1, var0_x1, kernel, x1, Z, Xt, Yt, noise, model, f_best, domain)
    #     x2_cand = x2_cand.reshape(1, -1)
    #
    #     print(result)
    #     x2_opt_list.append(x2_cand); opt_value_list.append(cand_value); grad_list.append(x2_cand_grad)
    #     x20_list.append(x20)
    #
    # index_opt= int(np.argmax(np.array(opt_value_list)))
    # x2_opt= x2_opt_list[index_opt]
    # opt_value= opt_value_list[index_opt]
    # x2_opt_grad= grad_list[index_opt]
    # x20_opt= x20_list[index_opt]

    x1_opt, opt_value_x1,  x1_opt_grad, result= \
                        two_opt_EI_optimize(f_best, Xt, Yt, model, domain, noise, kernel, num_inner_opt_restarts,
                                  num_outer_opt_restarts, monte_carlo_samples, D, Q=1)

    '''compare with grid'''
    x_grid1 = np.linspace(domain[0][0], domain[0][1], 21);

    X_grid = x_grid1.reshape(-1,1)

    if type(Z) != np.ndarray:
        Z_conv = np.atleast_2d(Z)
    else:
        Z_conv= Z.copy()

    two_opt_values= np.zeros([X_grid.shape[0], 1])
    for i in range(X_grid.shape[0]):
        x1_grid= (X_grid[i,:]).reshape(1,-1)
        u0_x1_grid, var0_x1_grid = model.predict_f(x1_grid); u0_x1_grid = u0_x1_grid.numpy(); var0_x1_grid = var0_x1_grid.numpy()

        sigma0_x1_grid_grid = np.sqrt(var0_x1_grid)

        two_opt= get_two_step_improvement_only(x1_grid, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)
        two_opt_values[i, 0]= two_opt


    # index_max= int(np.argmax(two_opt_values, axis=0))
    # x1_grid_opt= X_grid[index_max, :].reshape(1,-1)
    # grid_opt_value = two_opt_values[index_max, 0]

    '''calculate approximate derivative'''

    x_pl= x1_opt+0.0001; x_mn= x1_opt- 0.0001;
    x_pl_val= get_two_step_improvement_only(x_pl, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)
    x_mn_val= get_two_step_improvement_only(x_mn, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)
    x1_opt_val= get_two_step_improvement_only(x1_opt, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)

    app_deriv= ((x_pl_val- opt_value_x1 )/(x_pl- x1_opt)+\
                                    (opt_value_x1- x_mn_val)/(x1_opt- x_mn))/2

    print('x1_opt_val:{},opt_value_x1:{}'.format(x1_opt_val, opt_value_x1))
    print('app_deriv:\n{}\nopt_deriv:\n{}'.format(app_deriv, x1_opt_grad))

    '''comparison plots'''

    plt.figure()
    plt.plot(X_grid[:,0], two_opt_values[:,0], color= 'red', alpha= 0.5)
    plt.scatter(X_grid[:,0], two_opt_values[:,0], color= 'red', alpha= 0.5)
    plt.scatter(x1_opt, opt_value_x1, color= 'blue')
    # ax.scatter(x20_opt[0,0], x20_opt[0,1], np.zeros(1,), color='black', marker='X', label= 'initial x2- best optimum')

    plt.show()



def test_two_step_optimization_2d_branin():

    domain= [[-5, 10], [0, 15]]
    noise= (10**(-4))
    D=2
    num_inner_opt_restarts = 20#10*D
    num_outer_opt_restarts= 20#10*D

    Xt= np.random.uniform([domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], (3,2))
    Yt= branin(Xt)

    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt), kernel); model.likelihood.variance.assign(noise)

    x1= np.random.uniform([domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], (1,2))

    Z= np.random.normal(0.0, 1.0, [1,1])

    f_best= np.min(Yt,axis=0)

    u0_x1, var0_x1= model.predict_f(x1); u0_x1= u0_x1.numpy(); var0_x1= var0_x1.numpy()
    sigma0_x1= np.sqrt(var0_x1)

    x1_opt, opt_value_x1,  x1_opt_grad, result= \
                        two_opt_EI_optimize(f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts,
                                             num_outer_opt_restarts, D)


    '''calculate approximate derivative'''

    x1_pl= np.copy(x1_opt); x1_pl[0,0]+= 0.0001; x1_min= np.copy(x1_opt); x1_min[0,0]-= 0.0001
    x2_pl= np.copy(x1_opt); x2_pl[0,1]+= 0.0001; x2_min= np.copy(x1_opt); x2_min[0,1]-= 0.0001

    x1_pl_val= get_two_step_improvement_only(x1_pl, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)
    x1_min_val= get_two_step_improvement_only(x1_min, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)
    x2_pl_val= get_two_step_improvement_only(x2_pl, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)
    x2_min_val= get_two_step_improvement_only(x2_min, f_best, Xt, Yt, model, domain, Z, noise, kernel, num_inner_opt_restarts, D)

    app_deriv1= ((x1_pl_val[0]- opt_value_x1[0])/(x1_pl[0,0]-x1_opt[0,0])+\
                                    (opt_value_x1[0]- x1_min_val[0])/(x1_opt[0,0]- x1_min[0,0]))/2
    app_deriv2= ((x2_pl_val[0]- opt_value_x1[0])/(x2_pl[0,1]-x1_opt[0,1])+\
                                    (opt_value_x1[0]- x2_min_val[0])/(x2_opt[0,1]- x2_min[0,1]))/2

    app_deriv= np.array([app_deriv1, app_deriv2])
    print('app_deriv:\n{}\nx2_opt_deriv:\n{}'.format(app_deriv, x1_opt_grad))

    '''comparison plots'''

    plt.figure()
    plt.plot(X_grid[:,0], two_opt_values[:,0], color= 'red', alpha= 0.5)
    plt.scatter(X_grid[:,0], two_opt_values[:,0], color= 'red', alpha= 0.5)
    plt.scatter(x1_opt, opt_value_x1, color= 'blue')
    # ax.scatter(x20_opt[0,0], x20_opt[0,1], np.zeros(1,), color='black', marker='X', label= 'initial x2- best optimum')

    plt.show()