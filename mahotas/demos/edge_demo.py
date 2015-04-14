import mahotas
import numpy as np
from pylab import imshow, gray, show, subplot
from os import path

lena_image = path.join(
                    path.dirname(path.abspath(__file__)),
                    'data',
                    'lena.jpg')

photo = mahotas.imread(lena_image, as_grey=True)
photo = photo.astype(np.uint8)

gray()
subplot(131)
imshow(photo)

edge_sobel = mahotas.sobel(photo)
subplot(132)
imshow(edge_sobel)

edge_dog = mahotas.dog(photo)
subplot(133)
imshow(edge_dog)
show()
