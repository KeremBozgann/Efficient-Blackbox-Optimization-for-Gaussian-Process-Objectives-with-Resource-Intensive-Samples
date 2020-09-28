import numpy as np
import gpflow as gp
import GPy
import  matplotlib.pyplot as plt

num_samp= 10
# X = (2 * np.pi) * np.random.random(num_samp) - np.pi
# Y = np.sin(X) + np.random.normal(0, 0.2, num_samp)
# Y = np.array([np.power(abs(y), float(1) / 3) * (1, -1)[y < 0] for y in Y])
# X = X[:, None]
# Y = Y[:, None]
X = (2 * np.pi) * np.random.random(num_samp) - np.pi
Y = np.sin(X) + np.random.normal(0, 0.2, num_samp)
Y = np.array([np.exp(y) for y in Y])
X = X[:, None]
Y = Y[:, None]

Xt = (2 * np.pi) * np.random.random(100) - np.pi
Yt = np.sin(Xt) + np.random.normal(0, 0.2, 100)
Yt = np.array([np.exp(y) for y in Yt])
Xt = Xt[:, None]
Yt = Yt[:, None]

max_iters=100

warp_k = GPy.kern.RBF(1)
# warp_f = GPy.util.warping_functions.TanhFunction(n_terms=2)
warp_f = GPy.util.warping_functions.LogFunction()

warp_m = GPy.models.WarpedGP(X, Y, kernel=warp_k, warping_function=warp_f)
# warp_m['.*\.d'].constrain_fixed(1.0)

m = GPy.models.GPRegression(X, Y)

m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=max_iters)
warp_m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=max_iters)

y_pred, _= m.predict(Xt)
y_pred_warped, __ = warp_m.predict(Xt)

# plt.figure()
# plt.scatter(Xt, y_pred, color='red', label= 'GP')
# plt.scatter(Xt, y_pred_warped, color= 'blue', label= 'warped_GP')
# plt.scatter(Xt, Yt, color= 'green', label= 'True output')
# plt.legend()
# plt.show()

print(warp_m)
# print(warp_m['.*warp.*'])

warp_m.predict_in_warped_space = False
warp_m.plot(title="Warped GP - Latent space")
warp_m.predict_in_warped_space = True
warp_m.plot(title="Warped GP - Warped space")
m.plot(title="Standard GP")
warp_m.plot_warping()
plt.show()