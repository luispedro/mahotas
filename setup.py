import setuptools
from numpy.distutils.core import setup, Extension

morph_src = map(lambda F: 'morph/'+F,['_morph.cpp'])

morph = Extension('morph._morph', sources = morph_src, extra_compile_args=['-Wno-sign-compare'])

setup (name = 'pymorph++',
       version = '0.0.2-git',
       description = 'C++ implementation of morphological operations',
       packages=setuptools.find_packages(),
       ext_modules = [morph],
       )
