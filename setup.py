# -*- coding: utf-8 -*-
# Copyright (C) 2009-2020, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import division
try:
    import setuptools
except ImportError:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    from sys import exit
    exit(1)
import os
try:
    import numpy
except ImportError:
    class FakeNumpy(object):
        def get_include(self):
           return []
    numpy = FakeNumpy()


from distutils.command.build_ext import build_ext

exec(compile(open('mahotas/mahotas_version.py').read(),
             'mahotas/mahotas_version.py', 'exec'))

try:
    long_description = open('README.md', encoding='utf-8').read()
except:
    long_description = open('README.md').read()

undef_macros = []
define_macros = []
if os.environ.get('DEBUG'):
    undef_macros = ['NDEBUG']
    if os.environ.get('DEBUG') == '2':
        define_macros = [('_GLIBCXX_DEBUG','1')]

define_macros.append(('NPY_NO_DEPRECATED_API','NPY_1_7_API_VERSION'))
define_macros.append(('PY_ARRAY_UNIQUE_SYMBOL','Mahotas_PyArray_API_Symbol'))

extensions = {
    'mahotas._bbox': ['mahotas/_bbox.cpp'],
    'mahotas._center_of_mass': ['mahotas/_center_of_mass.cpp'],
    'mahotas._convex': ['mahotas/_convex.cpp'],
    'mahotas._convolve': ['mahotas/_convolve.cpp', 'mahotas/_filters.cpp'],
    'mahotas._distance': ['mahotas/_distance.cpp'],
    'mahotas._histogram': ['mahotas/_histogram.cpp'],
    'mahotas._interpolate': ['mahotas/_interpolate.cpp', 'mahotas/_filters.cpp'],
    'mahotas._labeled': ['mahotas/_labeled.cpp', 'mahotas/_filters.cpp'],
    'mahotas._morph': ['mahotas/_morph.cpp', 'mahotas/_filters.cpp'],
    'mahotas._thin': ['mahotas/_thin.cpp'],

    'mahotas.features._lbp': ['mahotas/features/_lbp.cpp'],
    'mahotas.features._surf': ['mahotas/features/_surf.cpp'],
    'mahotas.features._texture': ['mahotas/features/_texture.cpp', 'mahotas/_filters.cpp'],
    'mahotas.features._zernike': ['mahotas/features/_zernike.cpp'],
}

ext_modules = [setuptools.Extension(key, sources=sources, undef_macros=undef_macros, define_macros=define_macros, include_dirs=[numpy.get_include()]) for key,sources in extensions.items()]

packages = setuptools.find_packages()

package_dir = {
    'mahotas.tests': 'mahotas/tests',
    'mahotas.demos': 'mahotas/demos',
    }
package_data = {
    'mahotas.tests': ['data/*'],
    'mahotas.demos': ['data/*'],
    }

install_requires = open('requirements.txt').read().strip().split('\n')

tests_require = open('tests-requirements.txt').read().strip().split('\n')

copt={
    'msvc': ['/EHsc'], 
    'intelw': ['/EHsc']  
}

class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
           for e in self.extensions:
               e.extra_compile_args = copt[c]
        build_ext.build_extensions(self)

classifiers = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'Topic :: Scientific/Engineering :: Image Recognition',
'Topic :: Software Development :: Libraries',
'Programming Language :: Python',
'Programming Language :: Python :: 2',
'Programming Language :: Python :: 2.7',
'Programming Language :: Python :: 3',
'Programming Language :: Python :: 3.3',
'Programming Language :: Python :: 3.4',
'Programming Language :: Python :: 3.5',
'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7',
'Programming Language :: C++',
'Operating System :: OS Independent',
'License :: OSI Approved :: MIT License',
]

setuptools.setup(name = 'mahotas',
      version = __version__,
      description = 'Mahotas: Computer Vision Library',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      author = 'Luis Pedro Coelho',
      author_email = 'luis@luispedro.org',
      license = 'MIT',
      platforms = ['Any'],
      classifiers = classifiers,
      url = 'http://luispedro.org/software/mahotas',
      packages = packages,
      ext_modules = ext_modules,
      package_dir = package_dir,
      package_data = package_data,
      entry_points={
          'console_scripts': [
              'mahotas-features = mahotas.features_cli:main',
          ],
      },
      test_suite = 'nose.collector',
      install_requires = install_requires,
      tests_require = tests_require,
      cmdclass = {'build_ext': build_ext_subclass}
      )

