def image_path(name):
    from os import path
    import mahotas as mh
    return path.join(path.abspath(path.dirname(__file__)),
                'data',
                name)


def nuclear_image():
    import mahotas as mh
    return mh.imread(image_path('nuclear.png'))
