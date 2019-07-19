import mahotas
import numpy as np
import matplotlib.pyplot as plt
import os

plt.subplot(3,2,1)
f = mahotas.demos.load('nuclear')
f = f[:,:,0]
plt.title('input image, first channel')
plt.imshow(f)


plt.subplot(3,2,2)
f = mahotas.gaussian_filter(f, 4)
f = (f> f.mean())
plt.title('gaussian_filter')
plt.imshow(f)


plt.subplot(3,2,3)
labeled, n_nucleus  = mahotas.label(f)
plt.title('Found {} nuclei.'.format(n_nucleus))
plt.imshow(labeled)


plt.subplot(3,2,4)
sizes = mahotas.labeled.labeled_size(labeled)
too_big = np.where(sizes > 10000)
labeled = mahotas.labeled.remove_regions(labeled, too_big)
plt.title('remove_regions')
plt.imshow(labeled)


plt.subplot(3,2,5)
labeled = mahotas.labeled.remove_bordering(labeled)
plt.title('remove_bordering')
plt.imshow(labeled)


plt.subplot(3,2,6)
relabeled, n_left = mahotas.labeled.relabel(labeled)
plt.title('After filtering and relabeling, there are {} nuclei left.'.format(n_left))
plt.imshow(relabeled)


plt.tight_layout()
plt.show()
