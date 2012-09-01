# -*- coding: utf-8 -*-
# Copyright (C) 2009-2012, Luis Pedro Coelho <luis@luispedro.org>
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
except:
    print('''
setuptools not found.

On linux, the package is often called python-setuptools''')
    from sys import exit
    exit(1)
import os
import numpy.distutils.core as numpyutils


exec(compile(open('mahotas/mahotas_version.py').read(),
             'mahotas/mahotas_version.py', 'exec'))

long_description = open('docs/source/readme.rst').read()

undef_macros = []
define_macros = []
if os.environ.get('DEBUG'):
    undef_macros = ['NDEBUG']
    if os.environ.get('DEBUG') == '2':
        define_macros = [('_GLIBCXX_DEBUG','1')]


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

ext_modules = [numpyutils.Extension(key, sources=sources, undef_macros=undef_macros, define_macros=define_macros) for key,sources in extensions.items()]

packages = setuptools.find_packages()

package_dir = {
    'mahotas.tests': 'mahotas/tests',
    'mahotas.demos': 'mahotas/demos',
    }
package_data = {
    'mahotas.tests': ['data/*'],
    'mahotas.demos': ['data/*'],
    }

classifiers = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'Topic :: Scientific/Engineering :: Image Recognition',
'Topic :: Software Development :: Libraries',
'Programming Language :: Python',
'Programming Language :: C++',
'License :: OSI Approved :: MIT License',
]

numpyutils.setup(name = 'mahotas',
      version = __version__,
      description = 'Mahotas: Computer Vision Library',
      long_description = long_description,
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
      test_suite = 'nose.collector',
      )

