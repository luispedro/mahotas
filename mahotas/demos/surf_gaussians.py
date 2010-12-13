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
spoints = mahotas.surf.surf(f, 6, 24, 1)

f2 = mahotas.surf.show_surf(f, spoints)
imshow(f2)
show()
