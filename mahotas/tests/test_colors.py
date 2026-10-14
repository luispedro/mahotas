import mahotas
import numpy as np
import pytest
from mahotas.tests.utils import luispedro_jpg
from mahotas.colors import rgb2xyz, rgb2lab, xyz2rgb, rgb2grey, rgb2sepia

def test_colors():
    f = luispedro_jpg()
    lab = rgb2lab(f)
    assert np.max(np.abs(lab)) <= 100.
    assert np.max(np.abs(xyz2rgb(rgb2xyz(f)) - f)) < 1.

    lab8 = rgb2lab(f, dtype=np.uint8)
    assert lab.dtype != np.uint8
    assert lab8.dtype == np.uint8

    xyz = rgb2xyz(f, dtype=np.uint8)
    assert xyz.shape == f.shape
    assert xyz.dtype == np.uint8


def test_rgb2grey():
    f = luispedro_jpg()
    fg = rgb2grey(f)
    fg8 = rgb2grey(f, dtype=np.uint8)
    assert f.ndim == 3
    assert fg.ndim == 2
    assert fg8.ndim == 2
    assert fg.shape[0] == f.shape[0]
    assert fg.shape[1] == f.shape[1]
    assert fg.shape == fg8.shape
    assert fg8.dtype == np.uint8


def test_sepia():
    f = luispedro_jpg()
    sepia= mahotas.colors.rgb2sepia(f)
    assert sepia.shape == f.shape


def test_rgb2xyz_known_values():
    # sRGB white (255,255,255) should map to D65 illuminant XYZ ~ (0.9505, 1.0, 1.0890)
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    xyz = rgb2xyz(white)
    assert xyz.shape == (1, 1, 3)
    np.testing.assert_allclose(xyz[0, 0, 1], 1.0, atol=0.01,
        err_msg='XYZ Y channel for white should be ~1.0')
    np.testing.assert_allclose(xyz[0, 0, 0], 0.9505, atol=0.01,
        err_msg='XYZ X channel for white should be ~0.95')

    # sRGB mid-gray: linear = ((128/255)+0.055)/1.055)^2.4 ~ 0.2159
    gray = np.array([[[128, 128, 128]]], dtype=np.uint8)
    xyz_gray = rgb2xyz(gray)
    expected_linear = ((128/255. + 0.055)/1.055)**2.4
    # Y should equal the linear value (since R=G=B, Y = 0.2126*L + 0.7152*L + 0.0722*L = L)
    np.testing.assert_allclose(xyz_gray[0, 0, 1], expected_linear, atol=0.01,
        err_msg='XYZ Y for gray128 should use gamma decompression, not linear/12.92')


def test_xyz2rgb_known_values():
    # D65 white XYZ -> sRGB should give ~(255, 255, 255)
    white_xyz = np.array([[[0.9505, 1.0, 1.0890]]])
    rgb = xyz2rgb(white_xyz)
    np.testing.assert_allclose(rgb[0, 0], [255., 255., 255.], atol=2.0,
        err_msg='D65 white XYZ should map to sRGB ~(255,255,255)')


def test_rgb2lab_white():
    # Pure white in sRGB should give L*=100 in CIELAB
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    lab = rgb2lab(white)
    np.testing.assert_allclose(lab[0, 0, 0], 100., atol=1.0,
        err_msg='L* for white should be ~100')
    np.testing.assert_allclose(lab[0, 0, 1], 0., atol=1.0,
        err_msg='a* for achromatic white should be ~0')
    np.testing.assert_allclose(lab[0, 0, 2], 0., atol=1.0,
        err_msg='b* for achromatic white should be ~0')


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_xyz2rgb_requested_dtype_is_final_output_dtype(dtype):
    xyz = np.array([[[0.3, 0.3, 0.3], [0.001, 0.001, 0.001]]])
    # The first pixel uses gamma encoding and the second the linear branch.
    # Apply the published XYZ-to-linear-sRGB matrix before either transfer.
    expected_linear = np.array([[[0.36144, 0.28452, 0.27261],
                                 [0.0012048, 0.0009484, 0.0009087]]])
    expected = np.where(expected_linear <= 0.0031308,
                        12.92 * expected_linear,
                        1.055 * expected_linear ** (1.0 / 2.4) - 0.055) * 255
    actual = xyz2rgb(xyz, dtype=dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(actual, expected.astype(dtype), rtol=1e-6)
