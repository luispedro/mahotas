import numpy as np
import mahotas as mh
def test_find():
    for _ in range(16):
        f = np.random.random((128,128))
        c0,c1 = 43,23
        for h,w in [(12,56),
                    (11,7),
                    (12,7)]:
            matches = mh.find(f, f[c0:c0+h, c1:c1+w])
            coords = np.array(np.where(matches))
            assert np.all(coords.T == np.array((c0,c1)), 1).any()

def test_negative():
    f = 255*np.random.random((228,228))
    f = f.astype(np.uint8)
    h,w = 12,6
    t = f[:h,:w]
    matches = mh.find(f, t)
    coords = np.array(np.where(matches))
    for y,x in zip(*coords):
        if y < 0 or x < 0:
            continue
        assert np.all(f[y:y+h, x:x+w] == t)
