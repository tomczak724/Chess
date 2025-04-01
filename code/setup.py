# setup.py
# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("playground_cython.pyx", compiler_directives={"language_level": "3"})
)
