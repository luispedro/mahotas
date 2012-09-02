import numpy as np
import mahotas
import mahotas.convolve
from mahotas.convolve import convolve1d, gaussian_filter
import mahotas._filters
from os import path
from nose.tools import raises

def test_compare_w_ndimage():
    from scipy import ndimage
    A = np.arange(34*340, dtype='float64').reshape((34,340))%3
    B = np.ones((3,3), A.dtype)
    for mode in mahotas._filters.modes:
        assert np.all(mahotas.convolve(A, B, mode=mode) == ndimage.convolve(A, B, mode=mode))

def test_22():
    A = np.arange(1024).reshape((32,32))
    B = np.array([
        [0,1],
        [2,3],
        ])
    C = np.array([
        [0,1,0],
        [2,3,0],
        [0,0,0],
        ])
    AB = mahotas.convolve(A,B)
    AC = mahotas.convolve(A,C)
    assert AB.shape == AC.shape
    assert np.all(AB == AC)


@raises(ValueError)
def test_mismatched_dims():
    f = np.arange(128*128, dtype=float).reshape((128,128))
    filter = np.arange(17,dtype=float)-8
    filter **= 2
    filter /= -16
    np.exp(filter,filter)
    mahotas.convolve(f,filter)

def test_convolve1d():
    f = np.arange(64*4).reshape((16,-1))
    n = [.5,1.,.5]
    for axis in (0,1):
        g = convolve1d(f, n, axis)
        assert g.shape == f.shape

@raises(ValueError)
def test_convolve1d_2d():
    f = np.arange(64*4).reshape((16,-1))
    n = np.array([[.5,1.,.5],[0.,2.,0.]])
    convolve1d(f, n, 0)

luispedro_jpg = lambda: mahotas.imread(path.join(
    path.abspath(path.dirname(__file__)),
                '..',
                'demos',
                'data',
                'luispedro.jpg'), 1)


def test_gaussian_filter():
    from scipy import ndimage
    f = luispedro_jpg()
    for s in (4.,8.,12.):
        g = gaussian_filter(f, s)
        n = ndimage.gaussian_filter(f, s)
        assert np.max(np.abs(n - g)) < 1.e-5

def test_gaussian_order():
    im = np.arange(64*64).reshape((64,64))
    for order in (1,2,3):
        g_mat = mahotas.gaussian_filter(im, 2., order=order)

def test_gaussian_order_high():
    im = np.arange(64*64).reshape((64,64))
    @raises(ValueError)
    def gaussian_order(order):
        mahotas.gaussian_filter(im, 2., order=order)
    yield gaussian_order, 4
    yield gaussian_order, 5
    yield gaussian_order, -3
    yield gaussian_order, -1
    yield gaussian_order, 1.5

def test_haar():
    image = luispedro_jpg()
    image = image[:256,:256]
    wav = mahotas.haar(image)

    assert wav.shape == image.shape
    assert np.allclose((image[0].reshape((-1,2)).mean(1)+image[1].reshape((-1,2)).mean(1))/2, wav[0,:128]/2.)
    assert np.abs(np.mean(image**2) - np.mean(wav**2)) < 1.

    image = luispedro_jpg()
    wav =  mahotas.haar(image, preserve_energy=False)
    assert np.abs(np.mean(image**2) - np.mean(wav**2)) > 16.
    wav =  mahotas.haar(image, inline=True)
    assert id(image) == id(wav)

def test_ihaar():
    image = luispedro_jpg()
    image = image[:256,:256]
    wav = mahotas.haar(image)
    iwav = mahotas.ihaar(wav)
    assert np.allclose(image, iwav)
    iwav = mahotas.ihaar(wav, preserve_energy=False)
    assert not np.allclose(wav, iwav)
    iwav =  mahotas.ihaar(wav, inline=True)
    assert id(iwav) == id(wav)


def test_daubechies_D2_haar():
    image = luispedro_jpg()
    image = image[:256,:256]
    wav = mahotas.haar(image, preserve_energy=False)
    dau = mahotas.daubechies(image, 'D2')

    assert wav.shape == dau.shape
    assert np.allclose(dau, wav)

def test_3d_wavelets_error():
    @raises(ValueError)
    def call_f(f):
        f(np.arange(4*4*4).reshape((4,4,4)))

    yield call_f, mahotas.haar
    yield call_f, mahotas.ihaar
    yield call_f, lambda im: mahotas.daubechies(im, 'D4')

def test_wavelets_inline():
    def inline(f):
        im = np.arange(16, dtype=float).reshape((4,4))
        t = f(im, inline=True)
        assert id(im) == id(t)

    yield inline, mahotas.haar
    yield inline, lambda im,inline: mahotas.daubechies(im, 'D4', inline=inline)

def test_wavelet_iwavelet():
    f = luispedro_jpg()
    f = f[:256,:256]
    fo = f.copy()
    D4 = np.array([0.6830127,  1.1830127,  0.3169873, -0.1830127], dtype=np.float32)
    D4_high = D4[::-1].copy()
    D4_high[1::2] *= -1
    f = f[34]
    low = np.convolve(f, D4)
    high = np.convolve(f,D4_high)
    low[::2] = 0
    high[::2] = 0
    rec = (np.convolve(high, D4_high[::-1])+np.convolve(low, D4[::-1]))
    rec /= 2
    f2 = np.array([f])
    mahotas._convolve.wavelet(f2,D4)

    hand = np.concatenate((low[3::2],high[3::2]))
    wav = f2.ravel()
    assert np.allclose(hand,wav)
    mahotas._convolve.iwavelet(f2,D4)
    assert np.allclose(rec[3:-3],f)
    assert np.allclose(f2.ravel()[3:-3],f[3:-3])

    
def test_daubechies_idaubechies():
    f = luispedro_jpg()
    f = f[:256,:256]
    fo = f.copy()

    d = mahotas.daubechies(f, 'D8')
    r = mahotas.idaubechies(d, 'D8')
    assert np.mean( (r[4:-4,4:-4] - fo[4:-4,4:-4])**2) < 1.


def _is_power2(x):
    if x in (0,1,2,4,8,16,32,64,128): return True
    if (x & 1) != 0: return False
    return _is_power2(x // 2)

def test_center_decenter():
    from mahotas import wavelet_decenter
    from mahotas import wavelet_center
    np.random.seed(12)
    for border in (0, 1, 17):
        f = np.random.rand(51,100)
        fc = wavelet_center(f, border=border)
        assert all(map(_is_power2, fc.shape))
        
        fd = wavelet_decenter(fc, f.shape, border=border)

        assert fd.shape == f.shape
        assert np.all(fd == f)


def test_center_border():
    from mahotas import wavelet_center
    np.random.seed(12)
    for border in (16, 24):
        f = np.random.rand(51,100)
        fc = wavelet_center(f, border=border)
        assert np.all(fc[:border] == 0)
        assert np.all(fc[-border:] == 0)
        assert np.all(fc.T[:border] == 0)
        assert np.all(fc.T[-border:] == 0)

def test_center_wavelet_iwavelet_decenter():
    from mahotas import wavelet_center, wavelet_decenter
    import mahotas
    import numpy as np

    f = luispedro_jpg()
    f = f[:100,:250]
    fo = f.copy()

    for wav in ('D2', 'D4', 'D6', 'D8', 'D10', 'D12', 'D16'):
        fc = mahotas.wavelet_center(fo, border=24)
        t = mahotas.daubechies(fc, wav)
        r = mahotas.idaubechies(t, wav)
        rd = mahotas.wavelet_decenter(r, fo.shape, border=24)
        assert np.allclose(fo, rd)

