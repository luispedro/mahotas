import numpy as np
import mahotas.surf
import mahotas._surf
import pymorph
import mahotas.polygon
from pylab import *

f = mahotas.imread('mahotas/demos/data/luispedro.jpg', as_grey=True)
f = f.astype(np.uint8)
spoints = mahotas._surf.surf(mahotas.surf.integral(f.copy()), 4, 6, 2)
print "Nr points:", len(spoints)
f2 = np.dstack([f,f,f])

def rotate(y,x, a):
    sa = np.sin(a)
    ca = np.cos(a)
    return (ca*x-sa*y, sa*x+ca*y)

try:
    import milk
    descrs = spoints[:,5:]
    k = 5
    values, _  =milk.kmeans(descrs, k)
    colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in xrange(k)])
except:
    values = np.zeros(100)
    colors = [(255,0,0)]

for p,ci in zip(spoints[:100], values):
    y = p[0]
    x = p[1]
    scale = p[2]
    angle = p[5]
    size = int(scale*10)
    y0 = int(y) - size//2
    x0 = int(x) - size//2
    x1 = x + size
    y1 = y + size
    def rotate_around((p0,p1),(c0,c1), a):
        d0 = p0-c0
        d1 = p1 - c1
        d0,d1 = rotate(d0,d1,a)
        return int(c0+d0), int(c1+d1)
    polygon = [(y0,x0), (y0,x1), (y1,x1), (y1,x0), (y0,x0)]
    polygon = [rotate_around(p, (y,x), angle) for p in polygon]
    for p0,p1 in zip(polygon[:-1], polygon[1:]):
        mahotas.polygon.line(p0,p1, f2, color=colors[ci])
imshow(f2)
show()
