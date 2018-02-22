import mahotas
from os import path
import numpy as np
from matplotlib import pyplot as plt

try:
    nuclear_path = path.join(
                    path.dirname(path.abspath(__file__)),
                    'data',
                    'nuclear.png')
except NameError:
    nuclear_path = path.join('data', 'nuclear.png')

nuclear = mahotas.imread(nuclear_path)
nuclear = nuclear[:,:,0]
nuclear = mahotas.gaussian_filter(nuclear, 1.)
threshed  = (nuclear > nuclear.mean())
distances = mahotas.stretch(mahotas.distance(threshed))
Bc = np.ones((9,9))

maxima = mahotas.morph.regmax(distances, Bc=Bc)
spots,n_spots = mahotas.label(maxima, Bc=Bc)
surface = (distances.max() - distances)
areas = mahotas.cwatershed(surface, spots)
areas *= threshed



import random
from matplotlib import colors as c
colors = [plt.jet(c) for c in range(0, 256, 4)]
random.shuffle(colors)
colors[0] = (0.,0.,0.,1.)
rmap = c.ListedColormap(colors)
plt.imshow(areas, cmap=rmap)
plt.show()
