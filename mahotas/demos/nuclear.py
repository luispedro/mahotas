import mahotas
import numpy as np
from pylab import imshow, show

f = mahotas.imread('mahotas/demos/data/nuclear.png')
f = f[:,:,0]
imshow(f)
show()

f = mahotas.gaussian_filter(f, 4)
f = (f> f.mean())
imshow(f)
show()

labeled, n_nucleus  = mahotas.label(f)
print('Found {} nuclei.'.format(n_nucleus))
imshow(labeled)
show()
sizes = mahotas.labeled.labeled_size(labeled)
too_big = np.where(sizes > 10000)
labeled = mahotas.labeled.remove_regions(labeled, too_big)
imshow(labeled)
show()

labeled = mahotas.labeled.remove_bordering(labeled)
imshow(labeled)
show()

relabeled, n_left = mahotas.labeled.relabel(labeled)
print('After filtering and relabeling, there are {} nuclei left.'.format(n_left))
imshow(relabeled)
show()
