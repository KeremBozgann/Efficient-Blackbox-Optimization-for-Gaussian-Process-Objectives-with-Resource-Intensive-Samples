import numpy as np
import gpflow as gp
import h5py

rang= 5; disc= 20

x1= np.linspace(0,rang,disc)-rang/2; x2= np.linspace(0,rang,disc)-rang/2
X1,X2= np.meshgrid(x1,x2); X1,X2= X1.flatten(), X2.flatten(); X1, X2= X1.reshape(-1,1), X2.reshape(-1,1)
X= np.append(X1,X2, axis=1)

kern= gp.kernels.RBF(); model= gp.models.GPR((np.zeros([0,2]), np.zeros([0,1])),kernel=kern)
y_sample = model.predict_f_samples(X)

# with h5py.File('./datasets/gp_sample.h5', 'w') as hf:
#     hf.create_dataset('X',data= X); hf.create_dataset('y_sample', data= y_sample.numpy())
#
# with h5py.File('./datasets/gp_sample.h5', 'r') as hf:
#     X= np.array(hf.get('X')); y= np.array(hf.get('y_sample'))

