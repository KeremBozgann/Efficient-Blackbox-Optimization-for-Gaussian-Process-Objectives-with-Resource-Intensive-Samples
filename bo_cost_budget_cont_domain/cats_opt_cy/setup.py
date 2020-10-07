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
setup(
    ext_modules = cythonize("for_loop_speed_test.pyx", annotate=True), include_dirs=[numpy.get_include()]
)
#command prompt for creating c files:
# python setup.py build_ext --inplace --compiler=msvc