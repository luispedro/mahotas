import mahotas.demos
from os import path

def test_image_path():
    assert path.exists(mahotas.demos.image_path('luispedro.jpg'))
    assert not path.exists(mahotas.demos.image_path('something-that-does-not-exist'))

def test_demo_scripts_independent_of_cwd(tmp_path, monkeypatch):
    # superpixels.py used to open a path relative to the working directory
    import runpy
    import warnings
    import pytest
    from mahotas.io import error_imread
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if mahotas.imread is error_imread:
        pytest.skip("Missing I/O backend")
    monkeypatch.chdir(tmp_path)
    demos_dir = path.dirname(mahotas.demos.__file__)
    for script in ['superpixels.py', 'thresholding.py', 'edge_demo.py', 'wavelet_compress.py']:
        with warnings.catch_warnings():
            # plt.show() warns with the non-interactive Agg backend
            warnings.simplefilter('ignore', UserWarning)
            runpy.run_path(path.join(demos_dir, script), run_name='__main__')
        plt.close('all')
