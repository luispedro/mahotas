import mahotas.demos
from os import path

def test_image_path():
    assert path.exists(mahotas.demos.image_path('luispedro.jpg'))
    assert not path.exists(mahotas.demos.image_path('something-that-does-not-exist'))

