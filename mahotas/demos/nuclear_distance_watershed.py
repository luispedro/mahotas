import mahotas as mh
from os import path
import numpy as np
from matplotlib import pyplot as plt

nuclear = mh.demos.load('nuclear')
nuclear = nuclear[:,:,0]
nuclear = mh.gaussian_filter(nuclear, 1.)
threshed  = (nuclear > nuclear.mean())
distances = mh.stretch(mh.distance(threshed))
Bc = np.ones((9,9))

maxima = mh.morph.regmax(distances, Bc=Bc)
spots,n_spots = mh.label(maxima, Bc=Bc)
surface = (distances.max() - distances)
areas = mh.cwatershed(surface, spots)
areas *= threshed



import random
from matplotlib import colors
from matplotlib import cm
cols = [cm.jet(c) for c in range(0, 256, 4)]
random.shuffle(cols)
cols[0] = (0.,0.,0.,1.)
rmap = colors.ListedColormap(cols)
plt.imshow(areas, cmap=rmap)
plt.show()
