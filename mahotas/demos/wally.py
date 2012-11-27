from pylab import imshow
import mahotas
wally = mahotas.imread('data/DepartmentStore.jpg')
wfloat = wally.astype(float)
r,g,b = wfloat.transpose((2,0,1))
w = wfloat.mean(2)
pattern = np.ones((24,16), float)
for i in xrange(2):
    pattern[i::4] = -1
v = mahotas.convolve(r-w, pattern)
mask = (v == v.max())
mask = mahotas.dilate(mask, np.ones((48,24)))
wally -= .8*wally * ~mask[:,:,None]
imshow(wally)


