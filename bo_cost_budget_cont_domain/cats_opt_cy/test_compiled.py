import test_cy
# from test_cy import naive_convolve
# import numpy as np
# naive_convolve(np.random.randint(10, size=(1,3)),np.random.randint(10, size=(1,3)))
import time
def fun_for_(n):
    y=0
    for i in range(n):
        y+= i**2
    return y

import for_loop_speed_test

t1_cy= time.clock()
print(for_loop_speed_test.fun_for(1500))
t2_cy= time.clock()

t1_py= time.clock()
print(fun_for_(1500))
t2_py= time.clock()

print('cy_time:{}, py_time:{}'.format((t2_cy- t1_cy), (t2_py- t1_py)))
