import h5py

import numpy as np

with h5py.File('./test_h5.h5', 'a') as hf:
    hf.create_dataset('test1', data= np.random.rand(10,2))

with h5py.File('./test_h5.h5', 'a') as hf:
    hf.create_dataset('test2', data= np.random.rand(10,2))

with h5py.File('./test_h5.h5', 'r') as hf:
    print(np.array(hf.get('test1')).shape)
    print(np.array(hf.get('test2')).shape)
