# -*- coding: utf-8 -*-
# Copyright (C) 2009-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

from __future__ import division
try:
    import setuptools
except:
    print '''
setuptools not found.

On linux, the package is often called python-setuptools'''
    from sys import exit
    exit(1)
import os
import numpy.distutils.core as numpyutils


execfile('mahotas/mahotas_version.py')
long_description = file('docs/source/readme.rst').read()

undef_macros=[]
if os.environ.get('DEBUG'):
    undef_macros=['NDEBUG']

extensions = {
    'mahotas._bbox': ['mahotas/_bbox.cpp'],
    'mahotas._center_of_mass': ['mahotas/_center_of_mass.cpp'],
    'mahotas._convex': ['mahotas/_convex.cpp'],
    'mahotas._convolve': ['mahotas/_convolve.cpp', 'mahotas/_filters.cpp'],
    'mahotas._distance': ['mahotas/_distance.cpp'],
    'mahotas._histogram': ['mahotas/_histogram.cpp'],
    'mahotas._interpolate': ['mahotas/_interpolate.cpp', 'mahotas/_filters.cpp'],
    'mahotas._labeled': ['mahotas/_labeled.cpp', 'mahotas/_filters.cpp'],
    'mahotas._lbp': ['mahotas/_lbp.cpp'],
    'mahotas._morph': ['mahotas/_morph.cpp', 'mahotas/_filters.cpp'],
    'mahotas._surf': ['mahotas/_surf.cpp'],
    'mahotas._texture': ['mahotas/_texture.cpp', 'mahotas/_filters.cpp'],
    'mahotas._thin': ['mahotas/_thin.cpp'],
    'mahotas._zernike': ['mahotas/_zernike.cpp'],
}

ext_modules = [numpyutils.Extension(key, sources=sources, undef_macros=undef_macros) for key,sources in extensions.iteritems()]

packages = setuptools.find_packages()

package_dir = {
    'mahotas.tests': 'mahotas/tests',
    }
package_data = {
    'mahotas.tests': ['data/*.png'],
    }

classifiers = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'Topic :: Scientific/Engineering :: Image Recognition',
'Topic :: Software Development :: Libraries',
'Programming Language :: Python',
'Programming Language :: C++',
'License :: OSI Approved :: GNU General Public License (GPL)',
]

numpyutils.setup(name = 'mahotas',
      version = __version__,
      description = 'Mahotas: Computer Vision Library',
      long_description = long_description,
      author = 'Luis Pedro Coelho',
      author_email = 'luis@luispedro.org',
      license = 'GPL',
      platforms = ['Any'],
      classifiers = classifiers,
      url = 'http://luispedro.org/software/mahotas',
      packages = packages,
      ext_modules = ext_modules,
      package_dir = package_dir,
      package_data = package_data,
      test_suite = 'nose.collector',
      )

