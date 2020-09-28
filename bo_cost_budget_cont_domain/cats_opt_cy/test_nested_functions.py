#
# def fun_nest(float x):
#
#     cdef float result
#     result= x**2+ y**2
#     return result
#
# def fun(float x, float y):
#     cdef float result
#     result = fun_nest(x)+y
#     return result

# def fun_nest(x, *args, *kwargs):
#     a = args[0];
#     b = args[1]
#     x1= kwargs[]
#     return x ** 2+ a+ b

def fun_nest(x):
    return x**2+y**2

def fun(x,y):
    return fun_nest(x)+y
