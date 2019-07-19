from __future__ import print_function
import mahotas as mh
from pylab import gray, imshow, show


luispedro = mh.demos.load('luispedro')
luispedro = luispedro.max(2)
T = mh.otsu(luispedro)
lpbin = (luispedro > T)
eye = lpbin[112:180,100:190]
gray()
imshow(eye)
show()
imshow(~mh.morph.close(~eye))
show()
imshow(~mh.morph.open(~eye))
show()
