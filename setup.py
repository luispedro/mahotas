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
import numpy.distutils.core as numpyutils
execfile('mahotas/mahotas_version.py')
long_description = '''\
=======
Mahotas
=======
Python Image Processing Toolkit
-------------------------------

Image Processing Library for Python.

It includes a couple of algorithms implemented in C++ for speed while operating
in numpy arrays.

Notable algorithms:
 - watershed.
 - thresholding.
 - convex points calculations.
 - hit & miss. thinning.
 - Zernike & Haralick features.
 - freeimage based image loading (requires freeimage libraries to be
   installed).


*Website*: `http://luispedro.org/software/mahotas
<http://luispedro.org/software/mahotas>`_

*API Docs*: `http://packages.python.org/mahotas/
<http://packages.python.org/mahotas/>`_

*Mailing List*: Use the `pythonvision mailing list
<http://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc.
'''

histogram = numpyutils.Extension('mahotas/_histogram', sources = ['mahotas/_histogram.cpp'])
morph = numpyutils.Extension('mahotas._morph', sources = ['mahotas/_morph.cpp'], extra_compile_args=['-Wno-sign-compare'])
bbox = numpyutils.Extension('mahotas/_bbox', sources = ['mahotas/_bbox.cpp'])
center_of_mass = numpyutils.Extension('mahotas/_center_of_mass', sources = ['mahotas/_center_of_mass.cpp'])
texture = numpyutils.Extension('mahotas/_texture', sources = ['mahotas/_texture.cpp'])
convex = numpyutils.Extension('mahotas/_convex', sources = ['mahotas/_convex.cpp'])
zernike = numpyutils.Extension('mahotas/_zernike', sources = ['mahotas/_zernike.cpp'])
distance = numpyutils.Extension('mahotas/_distance', sources = ['mahotas/_distance.cpp'])

ext_modules = [histogram, morph, bbox, center_of_mass, texture, convex, zernike, distance]

packages = setuptools.find_packages()
if 'tests' in packages: packages.remove('tests')

numpyutils.setup(name = 'mahotas',
      version = __version__,
      description = 'Mahotas: Python Image Processing Library',
      long_description = long_description,
      author = 'Luis Pedro Coelho',
      author_email = 'lpc@cmu.edu',
      url = 'http://luispedro.org/software/mahotas',
      packages = packages,
      ext_modules = ext_modules,
      )

