from __future__ import print_function
import mahotas
from pylab import gray, imshow, show
import numpy as np

luispedro = mahotas.imread('./data/luispedro.jpg')
luispedro = luispedro.max(2)
T = mahotas.otsu(luispedro)
lpbin = (luispedro > T)
eye = lpbin[112:180,100:190]
gray()
imshow(eye)
show()
imshow(~mahotas.morph.close(~eye))
show()
imshow(~mahotas.morph.open(~eye))
show()
