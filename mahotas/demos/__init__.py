'''Demo Module

Includes functions to load demo images:

wally
    Colour image of "Find Wally"    
lena
    The classic Lena image
luispedro
    A colour photograph
nuclear
    A fluorescent microscopy image of cell nuclei
'''
def image_path(name):
    from os import path
    import mahotas as mh
    return path.join(path.abspath(path.dirname(__file__)),
                'data',
                name)


def load(image_name, as_grey=None):
    '''
    Loads a demo image

    Parameters
    ----------
    image_name : str
        Name of one of the demo images
    as_grey : bool, optional
        Whether to convert to greyscale

    Returns
    -------
    im : ndarray
        Image
    '''
    from os import path
    import mahotas as mh
    _demo_images  = {
        'wally' : 'DepartmentStore.jpg',
        'departmentstore' : 'DepartmentStore.jpg',
        'lenna' : 'lena.jpg',
        'lena' : 'lena.jpg',
        'luispedro' : 'luispedro.jpg',
        'nuclear' : 'nuclear.png',
    }
    if image_name.lower() not in _demo_images:
        raise KeyError('mahotas.demos.load: Unknown demo image "{}", known images are {}'.format(image_name, list(_demo_images.keys())))

    image_name = image_path(_demo_images[image_name.lower()])
    return mh.imread(image_name, as_grey=as_grey)

def nuclear_image():
    '''
    Loads the nuclear example image

    Returns
    -------
    im : ndarray
        nuclear image
    '''
    return load('nuclear')
    
