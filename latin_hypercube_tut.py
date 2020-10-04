
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import *

import sys
sys.path.append('./bo_cost_budget_cont_domain')
from util import *
sys.path.append('./functions')
from branin_res import *
from mpl_toolkits.mplot3d import Axes3D

def latin_hypercube_2d_uniform(n):

    lower_limits= np.arange(float(0), float(n))/n
    upper_limits= np.arange(1.0, n+1)/(n)

    points= np.random.uniform(low= lower_limits, high= upper_limits, size= [2,n]).T
    np.random.shuffle(points[:,1])

    return points

points= latin_hypercube_2d_uniform(10)

plt.figure(figsize= (5,5))
plt.xlim([0,1]); plt.ylim([0,1])
plt.scatter(points[:,0], points[:,1])
plt.show()

def scale(domain, X):

    for i in range(len(domain)):
        lowi= domain[i][0]
        highi= domain[i][1]
        middle=( highi+lowi)/2
        X[:,i]= (X[:,i]-0.5)*(highi- lowi)+middle

    return X

def test_pydoe():


    X = lhs(2, samples=1000, criterion='maximin')
    domain= [[0,15], [-5, 3]]
    X_rand = np.random.uniform(low= [domain[0][0], domain[1][0]], high= [domain[0][1],domain[1][1]], size = (1000, 2))
    X= scale(domain, X)
    plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], color= 'green')
    plt.scatter(X_rand[:, 0], X_rand[:, 1], color= 'red')
    plt.show()

def exp_2d(X):
    X1= X[:,0].reshape(-1,1)
    X2= X[:, 1].reshape(-1,1)
    result= np.exp(-4*(X1**2+X2**2))
    return result

def exp_1d(x):

    result= np.exp(-32*(x-0.5)**2)
    return result

def cdf_exp_1d(X_grid):
    sum= np.sum(exp_1d(X_grid), axis= 0)
    cdf_list= np.empty([X_grid.shape[0], 1])
    for i in range(X_grid.shape[0]):
        # xi= X_grid[i, :].reshape(1,-1)
        sumi= np.sum(exp_1d(X_grid[0:i, :]),axis=0)
        cdf_list[i, 0]= sumi/sum

    return cdf_list

def find_closest_in_lookup(X_rand, look_up, X_look_up):

    inv_transform_list= np.empty([X_rand.shape[0], 1])
    for i in range(X_rand.shape[0]):
        xi= X_rand[i, :].reshape(1,-1)
        dist= (xi- look_up)**2
        index= np.argmin(dist, axis = 0)
        print('xi', xi)
        print('index', index)
        inv_transform_list[i, :]= X_look_up[index, :].reshape(1,-1)
    return inv_transform_list

def inverse_transform_with_grid():

    domain= [[0,  1]]
    X_look_up= create_grid(1001, domain)
    look_up= cdf_exp_1d(X_look_up)

    Y_grid= exp_1d(X_look_up)
    X_rand = np.random.uniform(low= [domain[0][0]], high= [domain[0][1]], size = (100, 1))
    inv_transform= find_closest_in_lookup(X_rand, look_up, X_look_up)
    Y_inv_transform= exp_1d(inv_transform)

    '''plot inverse transformation samplings'''
    plt.figure()
    plt.scatter(X_look_up, Y_grid, color= 'red', alpha= 0.2)
    plt.scatter(inv_transform, Y_inv_transform, color= 'green')
    plt.show()
    return inv_transform, Y_inv_transform

def test_approximation_random():
    domain= [[0,  1]]
    # X_rand = np.random.uniform(low= [domain[0][0]], high= [domain[0][1]], size = (100, 1))
    num_ei_samples= 100
    num_lthc_samples= 1000
    D= len(domain)
    X_lthc = lhs(D, samples=num_lthc_samples, criterion='maximin')
    X_lthc= scale(domain, X_lthc)

    ei_X_lthc= exp_1d(X_lthc)
    choice= np.random.choice(np.arange(X_lthc.shape[0]), num_ei_samples, p= ei_X_lthc[:, 0]/np.sum(ei_X_lthc, axis=0))

    X_ei= X_lthc[choice, :]; value_X_ei= ei_X_lthc[choice, :]

    index_above = np.where(value_X_ei > np.mean(value_X_ei))[0]
    ei_values_above = value_X_ei[index_above, :].reshape(-1, 1)
    X_ei_above = X_ei[index_above, :]

    X_grid= create_grid(1001, domain)
    Y_grid= exp_1d(X_grid)

    X_ei_inv, Y_ei_inv= inverse_transform_with_grid()


    plt.figure()
    # ax = Axes3D(fig)
    plt.title('')
    plt.plot(X_grid[:, 0], Y_grid[:,0], color= 'red', alpha= 0.5)
    # plt.scatter(X_ei[:, 0], value_X_ei[:,0], color= 'blue', alpha= 1)
    # plt.scatter(X_ei_above, ei_values_above, color= 'blue', alpha= 0.3)
    plt.scatter(X_ei_inv,  Y_ei_inv, color= 'green', alpha= 0.3, label= 'sampled points')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    plt.legend()
    plt.savefig('C:/Users/asus/Masaüstü/inverse_transform.pdf', dpi= 400)


    plt.show()

    plt.figure()
    # ax = Axes3D(fig)
    plt.plot(X_grid[:, 0], Y_grid[:,0], color= 'red', alpha= 0.5)
    plt.scatter(X_ei[:, 0], value_X_ei[:,0], color= 'blue', alpha= 0.3, label= 'sampled points')
    # plt.scatter(X_ei_above, ei_values_above, color= 'blue', alpha= 0.3)
    # plt.scatter(X_ei_inv,  Y_ei_inv, color= 'green', alpha= 0.3)
    plt.legend()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    plt.legend()
    plt.savefig('C:/Users/asus/Masaüstü/proposed_sampling.pdf')

    plt.show()