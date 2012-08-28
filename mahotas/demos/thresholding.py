import mahotas
import numpy as np
from pylab import imshow, gray, show, subplot
from os import path

luispedro_image = path.join(
                    path.dirname(path.abspath(__file__)),
                    'data',
                    'luispedro.jpg')

photo = mahotas.imread(luispedro_image, as_grey=True)
photo = photo.astype(np.uint8)

gray()
subplot(131)
imshow(photo)

T_otsu = mahotas.otsu(photo)
print(T_otsu)
subplot(132)
imshow(photo > T_otsu)

T_rc = mahotas.rc(photo)
print(T_rc)
subplot(133)
imshow(photo > T_rc)
show()

