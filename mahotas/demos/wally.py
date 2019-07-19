from pylab import imshow
import mahotas as mh
import numpy as np

wally = mh.demos.load('DepartmentStore')
wfloat = wally.astype(float)
r,g,b = wfloat.transpose((2,0,1))
w = wfloat.mean(2)
pattern = np.ones((24,16), float)
for i in range(2):
    pattern[i::4] = -1
v = mh.convolve(r-w, pattern)
mask = (v == v.max())
mask = mh.dilate(mask, np.ones((48,24)))
wally -= np.array(.8*wally * ~mask[:,:,None], dtype=wally.dtype)
imshow(wally)


