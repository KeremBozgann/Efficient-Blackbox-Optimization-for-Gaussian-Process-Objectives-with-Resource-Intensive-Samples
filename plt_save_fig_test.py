
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.scatter(np.arange(10), np.arange(10))

plt.legend()
plt.savefig('./Results/07_09_2020/test')
plt.show()