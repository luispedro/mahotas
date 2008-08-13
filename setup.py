from numpy.distutils.core import setup, Extension

morph_src = map(lambda F: 'morph/'+F,['_morph.cpp'])

morph = Extension('morph._morph', sources = morph_src, extra_compile_args=['-Wno-sign-compare'])

setup (name = 'PyMorph',
       version = '0.1',
       description = 'C++ implementation of morphological operations',
       ext_modules = [morph],
       )
