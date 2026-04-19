import numpy as np
import mahotas.convolve
from mahotas.convolve import template_match

def test_template_match():
    np.random.seed(33)
    A = 255*np.random.random((1024, 512))
    t = A[8:12,8:12]+1
    m = template_match(A, t)

    assert m[10,10] == 16
    # I had tried testing over the whole image, but that took too long.
    for i in range(100):
        y = np.random.randint(m.shape[0]-4)
        x = np.random.randint(m.shape[1]-4)
        assert np.allclose(m[y+2,x+2], np.sum( (A[y:y+4, x:x+4] - t) ** 2))


def test_find_at_last_position():
    from mahotas.convolve import find
    # Place a unique pattern at the bottom-right corner, the last valid position
    # for find2d. The off-by-one bug (< instead of <=) causes this to be missed.
    f = np.zeros((8, 8), dtype=np.uint8)
    template = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    # Place at (6,6) — the last position where a 2x2 template fits in an 8x8 image
    f[6:8, 6:8] = template
    result = find(f, template)
    coords = np.array(np.where(result)).T
    assert len(coords) == 1, f"Expected 1 match, got {len(coords)}"
    assert list(coords[0]) == [6, 6]


def test_find_at_last_row_and_col():
    from mahotas.convolve import find
    # Template match at the last valid row (but not last column) and vice versa
    f = np.zeros((10, 10), dtype=np.uint8)
    t = np.array([[5, 6, 7]], dtype=np.uint8)  # 1x3 template
    # Place at last valid row (row 9) and a middle column
    f[9, 3:6] = t
    result = find(f, t)
    coords = np.array(np.where(result)).T
    assert [9, 3] in coords.tolist()

    # Now test last valid column with a 3x1 template
    f2 = np.zeros((10, 10), dtype=np.uint8)
    t2 = np.array([[5], [6], [7]], dtype=np.uint8)
    f2[3:6, 9] = t2.flatten()
    result2 = find(f2, t2)
    coords2 = np.array(np.where(result2)).T
    assert [3, 9] in coords2.tolist()
