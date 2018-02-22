import random
import matplotlib
from matplotlib import pyplot as plt

import mahotas

f = mahotas.imread('mahotas/demos/data/luispedro.jpg')
colors = [plt.cm.jet(c) for c in  range(256)]
random.seed(23)
random.shuffle(colors)
cmap = matplotlib.colors.ListedColormap (colors)

segmented, _ = mahotas.segmentation.slic(f, 16)
plt.imshow(segmented, cmap=cmap)
plt.show()

segmented, _ = mahotas.segmentation.slic(f, 64)
plt.imshow(segmented, cmap=cmap)
plt.show()

segmented, _ = mahotas.segmentation.slic(f, 128)
plt.imshow(segmented, cmap=cmap)
plt.show()

