from numpy.distutils.core import setup, Extension

numpypp_src = map(lambda F: 'numpypp/'+F,['dilate.cpp'])

numpypp = Extension('numpypp.numpypp', sources = numpypp_src, extra_compile_args=['-Wno-sign-compare'])

setup (name = 'Numpy++',
       version = '0.1',
       description = 'Numpy++',
       ext_modules = [numpypp],
       )
