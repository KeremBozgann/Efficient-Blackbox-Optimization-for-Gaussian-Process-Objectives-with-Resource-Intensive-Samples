from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("hello_world", ["hello_world.pyx"])]
# )
from setuptools import setup
from Cython.Build import cythonize
import numpy
import scipy

setup(
    ext_modules = cythonize("cats_optimize_cy.pyx", annotate=True), include_dirs=[numpy.get_include(), scipy.get_include()]
)
#command prompt for creating c files:
# python setup.py build_ext --inplace --compiler=msvc

