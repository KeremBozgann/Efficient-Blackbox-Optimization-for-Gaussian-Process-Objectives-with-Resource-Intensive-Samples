
import matplotlib.pyplot as plt
import numpy as np

def z_func(x, y):
    return (1 - (x ** 2 + y ** 3)) * np.exp(-(x ** 2 + y ** 2) / 2)

x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = z_func(X, Y)

im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=(-3, 3, -3, 3))
cset = plt.contour(Z, np.arange(-1, 1.5, 0.2), linewidths=2,
                   cmap=plt.cm.Set2,
                   extent=(-3, 3, -3, 3))

plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)



plt.colorbar(im)

plt.show()

plt.show()