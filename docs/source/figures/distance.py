import numpy as np
import mahotas
from pylab import imshow, savefig
A = np.zeros((100,100), bool)
A[40:60] = 1
W = mahotas.thin(A)
D = mahotas.distance(~W)
imshow(D)
savefig('distance.png')

