import numpy as np
from mahotas import zernike
from mahotas.center_of_mass import center_of_mass
from math import atan2
from numpy import cos, sin, conjugate, pi, sqrt

def _slow_znl(X,Y,P,n,l):
    def _polar(r,theta):
        x = r * cos(theta)
        y = r * sin(theta)
        return 1*x+1j*y

    v = 0.+0.j
    def _factorial(n):
        if n == 0: return 1.
        return n * _factorial(n - 1)
    for x,y,p in zip(X,Y,P):
        Vnl = 0.
        for m in xrange( int( (n-l)//2 ) + 1 ):
              Vnl += (-1.)**m * _factorial(n-m) /  \
            ( _factorial(m) * _factorial((n - 2*m + l) // 2) * _factorial((n - 2*m - l) // 2) ) * \
            ( sqrt(x*x + y*y)**(n - 2*m) * _polar(1.0, l*atan2(y,x)) )
        v += p * conjugate(Vnl)
    v *= (n+1)/pi
    return v 

def _slow_zernike(img, D, radius):
    zvalues = []

    X,Y = np.where(img > 0)
    P = img[X,Y].ravel()
    cofx,cofy = center_of_mass(img)
    Xn = ( (X -cofx)/radius).ravel()
    Yn = ( (Y -cofx)/radius).ravel()
    k = (np.sqrt(Xn**2 + Yn**2) <= 1.)
    frac_center = np.array(P[k], np.double)/img.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    frac_center = frac_center.ravel()

    for n in xrange(D+1):
        for l in xrange(n+1):
            if (n-l)%2 == 0:
                z = _slow_znl(Xn, Yn, frac_center, float(n), float(l))
                zvalues.append(abs(z))
    return zvalues

def test_zernike():
    A = (np.arange(1024) % 14).reshape((32, 32))
    slow = zernike.zernike(A, 12, 8.)
    fast = zernike.zernike(A, 12, 8.)
    delta = np.array(slow) - fast
    assert np.abs(delta).max() < 0.001
