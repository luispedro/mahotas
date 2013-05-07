import mahotas as mh
import numpy as np
def luispedro_jpg(as_grey=False):
    from os import path
    path = path.join(
                path.abspath(path.dirname(__file__)),
                'data',
                'luispedro.npy')
    im = np.load(path)
    if as_grey:
        transform = np.array([0.30, 0.59, 0.11])
        return np.dot(im, transform)
    return im
