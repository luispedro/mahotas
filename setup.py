# -*- coding: utf-8 -*-
# Copyright (C) 2009-2010, Luis Pedro Coelho <lpc@cmu.edu>
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

bbox = numpyutils.Extension('mahotas._bbox', sources = ['mahotas/_bbox.cpp'])
center_of_mass = numpyutils.Extension('mahotas._center_of_mass', sources = ['mahotas/_center_of_mass.cpp'])
convex = numpyutils.Extension('mahotas._convex', sources = ['mahotas/_convex.cpp'])
convolve = numpyutils.Extension('mahotas._convolve', sources = ['mahotas/_convolve.cpp', 'mahotas/_filters.cpp'])
distance = numpyutils.Extension('mahotas._distance', sources = ['mahotas/_distance.cpp'])
histogram = numpyutils.Extension('mahotas._histogram', sources = ['mahotas/_histogram.cpp'])
labeled = numpyutils.Extension('mahotas._labeled', sources = ['mahotas/_labeled.cpp', 'mahotas/_filters.cpp'])
lbp = numpyutils.Extension('mahotas._lbp', sources = ['mahotas/_lbp.cpp'])
morph = numpyutils.Extension('mahotas._morph', sources = ['mahotas/_morph.cpp', 'mahotas/_filters.cpp'])
surf = numpyutils.Extension('mahotas._surf', sources = ['mahotas/_surf.cpp'], undef_macros=undef_macros)
texture = numpyutils.Extension('mahotas._texture', sources = ['mahotas/_texture.cpp', 'mahotas/_filters.cpp'])
thin = numpyutils.Extension('mahotas._thin', sources = ['mahotas/_thin.cpp'])
zernike = numpyutils.Extension('mahotas._zernike', sources = ['mahotas/_zernike.cpp'])

ext_modules = [bbox, center_of_mass, convex, convolve, distance, histogram, labeled, lbp, morph, surf, texture, thin, zernike]

packages = setuptools.find_packages()
if 'tests' in packages: packages.remove('tests')

classifiers = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'License :: OSI Approved :: GNU General Public License (GPL)',
'Programming Language :: C++',
'Topic :: Scientific/Engineering :: Image Recognition',
'Topic :: Software Development :: Libraries',
'Programming Language :: Python',
]

numpyutils.setup(name = 'mahotas',
      version = __version__,
      description = 'Mahotas: Python Image Processing Library',
      long_description = long_description,
      author = 'Luis Pedro Coelho',
      author_email = 'lpc@cmu.edu',
      license = 'GPL',
      platforms = ['Any'],
      classifiers = classifiers,
      url = 'http://luispedro.org/software/mahotas',
      packages = packages,
      ext_modules = ext_modules,
      )

