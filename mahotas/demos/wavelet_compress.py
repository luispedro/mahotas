from __future__ import print_function
import numpy as np
import mahotas
from mahotas.thresholding import soft_threshold
from matplotlib import pyplot as plt
from os import path

# Let us load the data first:
luispedro_image = path.join(
                    path.dirname(path.abspath(__file__)),
                    'data',
                    'luispedro.jpg')

f = mahotas.imread(luispedro_image, as_grey=True)
f = f[:256,:256]
plt.gray()
# Show the data:
plt.imshow(f)
print("Fraction of zeros in original image:", np.mean(f==0))

# A baseline compression method: save every other pixel and only high-order bits:
direct = f[::2,::2].copy()
direct /= 8
direct = direct.astype(np.uint8)
print("Fraction of zeros in original image (after division by 8):", np.mean(direct==0))
plt.imshow(direct)


# Transform using D8 Wavelet to obtain transformed image t:
t = mahotas.daubechies(f,'D8')
plt.imshow(t)

# Discard low-order bits:
t /= 8
t = t.astype(np.int8)
print("Fraction of zeros in transform (after division by 8):", np.mean(t==0))
plt.imshow(t)

# Let us look at what this looks like
r = mahotas.idaubechies(t, 'D8')
plt.imshow(r)

# Go further, discard small values in the transformed space:
tt = soft_threshold(t, 12)
print("Fraction of zeros in transform (after division by 8 & soft thresholding):", np.mean(tt==0))

# Let us look again at what we have:
rt = mahotas.idaubechies(tt, 'D8')
plt.imshow(rt)
