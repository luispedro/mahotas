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
from importlib import resources

_demo_images  = {
    'wally' : 'DepartmentStore.jpg',
    'departmentstore' : 'DepartmentStore.jpg',
    'lenna' : 'lena.jpg',
    'lena' : 'lena.jpg',
    'luispedro' : 'luispedro.jpg',
    'nuclear' : 'nuclear.png',
}

def _data_file(name):
    return resources.files(__name__) / 'data' / name

def image_path(name):
    '''
    Return the full path to a demo image file

    Parameters
    ----------
    name : str
        File name of the image (e.g., ``'nuclear.png'``)

    Returns
    -------
    path : str
        Absolute path to the file in the demo data directory. This assumes
        mahotas is installed as regular files (the normal case); use
        :func:`load` to read an image regardless of how it is installed.

    See Also
    --------
    load : function
        Load a demo image by name
    '''
    return str(_data_file(name))


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
    import mahotas as mh
    if image_name.lower() not in _demo_images:
        raise KeyError(f'mahotas.demos.load: Unknown demo image "{image_name}", known images are {list(_demo_images.keys())}')

    # The I/O backends need a real file name, which as_file guarantees
    with resources.as_file(_data_file(_demo_images[image_name.lower()])) as fname:
        return mh.imread(str(fname), as_grey=as_grey)

def nuclear_image():
    '''
    Loads the nuclear example image

    Returns
    -------
    im : ndarray
        nuclear image
    '''
    return load('nuclear')
    
