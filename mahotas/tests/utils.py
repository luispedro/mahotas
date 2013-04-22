import mahotas as mh
def luispedro_jpg(as_grey=False):
    from os import path
    return mh.imread(path.join(
        path.abspath(path.dirname(__file__)),
                '..',
                'demos',
                'data',
                'luispedro.jpg'), as_grey=as_grey)

