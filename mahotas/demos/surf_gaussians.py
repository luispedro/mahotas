import mahotas.polygon
from pylab import imshow, show
import numpy as np
import mahotas.surf
import mahotas._surf

f = np.zeros((1024,1024))
Y,X = np.indices(f.shape)
Y -= 768
X -= 768
f += 120*np.exp(-Y**2/2048.-X**2/480.)
Y += 512
X += 512
rho = .7
f += 120*np.exp(-1./( 2*(1-rho**2)) *( Y**2/32/32.+X**2/24/24. + 2*rho*X*Y/32./24.))
fi = mahotas.surf.integral(f.copy())
spoints = mahotas._surf.surf(fi, 6, 24, 1)

def rotate(y,x, a):
    sa = np.sin(a)
    ca = np.cos(a)
    return (ca*x-sa*y, sa*x+ca*y)
f2 = np.dstack([f,f,f])
for p in spoints:
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
        mahotas.polygon.line(p0,p1, f2, color=(255,0,0))

imshow(f2.astype(np.uint8))
show()
