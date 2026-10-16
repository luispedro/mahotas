import mahotas
import numpy as np
from pylab import imshow, gray, show, subplot

photo = mahotas.demos.load('lena', as_grey=True)
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
