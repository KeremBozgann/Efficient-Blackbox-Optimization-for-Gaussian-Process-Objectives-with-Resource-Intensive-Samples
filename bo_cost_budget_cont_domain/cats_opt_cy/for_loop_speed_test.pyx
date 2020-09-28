
def fun_for(int n):
    cdef int i
    cdef int y=0
    for i in range(n):
        y+= i**2

    return y