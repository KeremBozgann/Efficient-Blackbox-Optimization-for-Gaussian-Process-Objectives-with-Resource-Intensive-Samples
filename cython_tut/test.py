# import hello_world


'''pyximport: create on fly'''
# import pyximport; pyximport.install()
# import hello_world

from __future__ import print_function

def fibon(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()

'''compare c and py implementation times'''
import time
import fib

'''c time'''
t1_c= time.clock()
fib.fib(2000)
t2_c= time.clock()

'''py time'''
t1_py= time.clock()
fibon(2000)
t2_py= time.clock()

print('c_time:{}, py_time:{}'.format((t2_c-t1_c), (t2_py-t1_py)))


'''primes example'''


import numpy as np

def primes_py(nb_primes):

    p= np.arange(1000)
    if nb_primes > 1000:
        nb_primes = 1000

    len_p = 0  # The current number of elements in p.
    n = 2
    while len_p < nb_primes:
        # Is n prime?
        for i in p[:len_p]:
            if n % i == 0:
                break

        # If no break occurred in the loop, we have a prime.
        else:
            p[len_p] = n
            len_p += 1
        n += 1

    # Let's return the result in a python list:
    result_as_list  = [prime for prime in p[:len_p]]
    return result_as_list



import primes

'''c time'''
t1_c= time.clock()
prime_c= primes.primes(200)
t2_c= time.clock()


'''py time'''
t1_py= time.clock()
prime_py= primes_py(200)
t2_py= time.clock()

print('c_time:{}, py_time:{}'.format((t2_c-t1_c), (t2_py-t1_py)))

