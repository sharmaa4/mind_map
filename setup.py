from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cython_optimizations.pyx"),
    include_dirs=[numpy.get_include()],
    script_args=["build_ext", "--inplace"]
)
