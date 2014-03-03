from __future__ import print_function
import numpy as np
import mahotas as mh
from mahotas.features import surf
from pylab import *

from os import path

f = mh.demos.load('luispedro', as_grey=True)
f = f.astype(np.uint8)
spoints = surf.surf(f, 4, 6, 2)
print("Nr points:", len(spoints))

try:
    import milk
    descrs = spoints[:,5:]
    k = 5
    values, _  =milk.kmeans(descrs, k)
    colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in range(k)])
except:
    values = np.zeros(100)
    colors = np.array([(255,0,0)])

f2 = surf.show_surf(f, spoints[:100], values, colors)
imshow(f2)
show()
