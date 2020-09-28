
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import gpflow as gp

sys.path.append('./functions')


from branin import branin

def func(x):
    x= np.array(x); x= x.reshape(1,-1)
    x1= x[0,0]; x2= x[0, 1]
    y= (x1-1)**2+(x2-1)**2

    return y

x0= [0.5,0.5]
minimize(func, x0, bounds= [[-5,5], [-5,5]], method= 'L-BFGS-B')


def EI(sigma_x, u_x , f_best):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))


    return EI_x


def EI_optimize(model, f_best):

    def func(x):
        x= x.reshape(1,-1)
        u_x, var_x= model.predict_f(x); sigma_x= np.sqrt(var_x)
        gama_x = (f_best - u_x) / sigma_x
        fi_x = norm.cdf(gama_x)
        EI_x = float(sigma_x * (gama_x * fi_x + norm.pdf(gama_x)))
        print('func ei')
        print('EI_x', EI_x)
        return -EI_x

    domain= [[-5,10], [0,15]]
    x1= np.random.uniform(low= domain[0][0], high= domain[0][1])
    x2= np.random.uniform(low= domain[1][0], high= domain[1][1])

    x0= [x1,x2]

    Xt= np.random.uniform(low= [domain[0][0], domain[1][0]],
                          high= [domain[0][1], domain[1][1]], size=(10,2))

    Yt= branin(Xt)
    kernel= gp.kernels.RBF()
    model= gp.models.GPR((Xt, Yt) ,kernel)
    model.likelihood.variance.assign(10**(-4))

    f_best= float(np.max(Yt,axis=0))
    minimize(func, x0, bounds= domain, method= 'L-BFGS-B')


x1= np.random.uniform(low= domain[0][0], high= domain[0][1])
x2= np.random.uniform(low= domain[1][0], high= domain[1][1])

x0= [x1,x2]

domain= [[-5,10], [0,15]]

Xt= np.random.uniform(low= [domain[0][0], domain[1][0]],
                      high= [domain[0][1], domain[1][1]], size=(10,2))

Yt= branin(Xt)
kernel= gp.kernels.RBF()
model= gp.models.GPR((Xt, Yt) ,kernel)
model.likelihood.variance.assign(10**(-4))

rang= 10; disc= 20
x1 = np.linspace(domain[0][0], domain[0][1], disc)
x2 = np.linspace(domain[1][0], domain[1][1],  disc)
X1, X2 = np.meshgrid(x1, x2);
X1, X2 = X1.flatten(), X2.flatten();
X1, X2 = X1.reshape(-1, 1), X2.reshape(-1, 1)
X = np.append(X1, X2, axis=1)

u_X, var_X= model.predict_f(X); sigma_X= np.sqrt(var_X)
f_best= float(np.max(Yt,axis=0))

EI_X= EI(sigma_X, u_X, f_best)

index_opt= np.argmax(EI_X, axis=0); EI_opt= np.max(EI_X, axis=0)
x_opt= X[index_opt, :]
print('index_opt:{}, x_opt:{}, EI_opt:{}', index_opt, EI_opt, x_opt )
