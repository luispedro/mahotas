def image_path(name):
    from os import path
    import mahotas as mh
    return path.join(path.abspath(path.dirname(__file__)),
                'data',
                name)


def load(image_name):
    '''
    Loads a demo image

    Parameters
    ----------
    image_name : str
        Name of one of the demo images

    Returns
    -------
    im : ndarray
        Image
    '''
    from os import path
    import mahotas as mh
    _demo_images  = {
        'departmentstore' : 'DepartmentStore.jpg',
        'lena' : 'lena.jpg',
        'luispedro' : 'luispedro.jpg',
        'nuclear' : 'nuclear.png',
    }
    if image_name.lower() not in _demo_images:
        raise KeyError('mahotas.demos.load: Unknown demo image "{}", known images are {}'.format(image_name, list(_demo_images.keys())))

    image_name = image_path(_demo_images[image_name.lower()])
    return mh.imread(image_name)

def nuclear_image():
    '''
    Loads the nuclear example image

    Returns
    -------
    im : ndarray
        nuclear image
    '''
    return load('nuclear')
    
