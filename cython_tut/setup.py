from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("hello_world", ["hello_world.pyx"])]
# )
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("primes.pyx", annotate=True)
)
#command prompt for creating c files: python setup.py build_ext --inplace --compiler=msvc