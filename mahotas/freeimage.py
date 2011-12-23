import ctypes
import ctypes.util
import numpy as np
import sys
import os

_API = {
    'FreeImage_AllocateT':
                (ctypes.c_void_p,
                    [ ctypes.c_int # type
                    , ctypes.c_int # width
                    , ctypes.c_int # height
                    , ctypes.c_int #bpp
                    , ctypes.c_uint # red_mask
                    , ctypes.c_uint # green_mask
                    , ctypes.c_uint # blue_mask
                    ]),
    'FreeImage_Save': (ctypes.c_int,
                        [ ctypes.c_int # type
                        , ctypes.c_void_p # bitmap
                        , ctypes.c_char_p # filename
                        , ctypes.c_int # flags
                        ]),
    'FreeImage_SetOutputMessage':
                (None, [ctypes.c_void_p]), # callback
    'FreeImage_ConvertToGreyscale':
                (ctypes.c_void_p, # FIBITMAP * new_bitmap
                    [ctypes.c_void_p]), # FIBITMAP* bitmap
    'FreeImage_GetFIFFromFilename':
                (ctypes.c_int, # FREE_IMAGE_FORMAT
                    [ctypes.c_char_p]), # const char* filename
    'FreeImage_IsLittleEndian':
                (ctypes.c_int, # BOOL
                    []),
    'FreeImage_FIFSupportsExportBPP':
                (ctypes.c_int, # BOOL
                    [ctypes.c_int, # FREE_IMAGE_FORMAT format
                     ctypes.c_int]), # int bpp
    'FreeImage_FIFSupportsExportType':
                (ctypes.c_int, # BOOL
                    [ctypes.c_int # FREE_IMAGE_FORMAT fif
                    ,ctypes.c_int]), # FREE_IMAGE_TYPE type
    'FreeImage_Load': (ctypes.c_void_p,
                       [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]),
    'FreeImage_Unload': (None,
                        [ctypes.c_void_p]),
    'FreeImage_GetWidth': (ctypes.c_uint,
                           [ctypes.c_void_p]),
    'FreeImage_GetHeight': (ctypes.c_uint,
                           [ctypes.c_void_p]),
    'FreeImage_GetImageType': (ctypes.c_uint,
                               [ctypes.c_void_p]),
    'FreeImage_GetFileTypeFromMemory': (ctypes.c_int,
                                [ctypes.c_void_p, ctypes.c_int]),
    'FreeImage_GetFileType': (ctypes.c_int,
                                [ctypes.c_char_p, ctypes.c_int]),
    'FreeImage_GetBPP': (ctypes.c_uint,
                         [ctypes.c_void_p]),
    'FreeImage_GetPitch': (ctypes.c_uint,
                           [ctypes.c_void_p]),
    'FreeImage_OpenMultiBitmap' : (ctypes.c_void_p, # FIMULTIBITMAP*
                            [ctypes.c_int # FREE_IMAGE_FORMAT format
                            ,ctypes.c_char_p # filename
                            ,ctypes.c_int # BOOL create_new
                            ,ctypes.c_int # BOOL read_only
                            ,ctypes.c_int # BOOL keep_cache_in_memory
                            ,ctypes.c_int]), # int flags
    'FreeImage_GetPageCount' : (ctypes.c_int, [ctypes.c_void_p]),
    'FreeImage_AppendPage' : (None,
                                [ctypes.c_void_p # FIMULTIBITMAP*
                                ,ctypes.c_void_p]), # BITMAP
    'FreeImage_LockPage' : (ctypes.c_void_p, # FIBITMAP*
                                [ctypes.c_void_p # FIMULTIBITMAP
                                ,ctypes.c_int]), # int page
    'FreeImage_UnlockPage' : (None,
                                [ctypes.c_void_p # FIMULTIBITMAP*
                                ,ctypes.c_void_p # FIBITMAP* data
                                ,ctypes.c_int]), # BOOL changed
    'FreeImage_CloseMultiBitmap' : (ctypes.c_int, # BOOL
                            [ctypes.c_void_p, # FIMULTIBITMAP* bitmap
                             ctypes.c_int]), # int flags
    'FreeImage_GetBits': (ctypes.c_void_p,
                          [ctypes.c_void_p]),
    'FreeImage_OpenMemory': (ctypes.c_void_p,
                            [ctypes.c_void_p, ctypes.c_uint32]),
    'FreeImage_AcquireMemory': (ctypes.c_int,
                            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)]),
    'FreeImage_CloseMemory': (None,
                            [ctypes.c_void_p]),
    'FreeImage_LoadFromMemory': (ctypes.c_void_p,
                            [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]),
    'FreeImage_SaveToMemory': (ctypes.c_int,
                            [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
    }

class _ctypes_wrapper(object):
    pass
# Albert's ctypes pattern
def _register_api(lib, api):
    nlib = _ctypes_wrapper()
    for f, (restype, argtypes) in api.iteritems():
        func = getattr(lib, f)
        func.restype = restype
        func.argtypes = argtypes
        setattr(nlib, f, func)
    return nlib

libname = ctypes.util.find_library('freeimage')
if libname:
    _FI = ctypes.CDLL(libname)
else:
    _FI = None
    _lib_dirs = os.environ.get('LD_LIBRARY_PATH','').split(':')
    _lib_dirs = filter(None, _lib_dirs)
    _lib_dirs.extend([
        os.path.dirname(__file__),
        '/lib',
        '/usr/lib',
        '/usr/local/lib',
        '/opt/local/lib',
        ])
    _possible_filenames = (
        'libfreeimage',
        'libFreeImage',
        )
    if sys.platform == 'win32':
        _FI = ctypes.windll.LoadLibrary(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), 'FreeImage.dll'))
        if not _FI:
            raise OSError('mahotas.freeimage: could not find FreeImage.dll')
    else:
        for d in _lib_dirs:
            for libname in _possible_filenames:
                try:
                    _FI = np.ctypeslib.load_library(libname, d)
                except OSError:
                    pass
                else:
                    break

            if _FI is not None:
                break

        if not _FI:
            raise OSError('mahotas.freeimage: could not find libFreeImage in any of the following '
                          'directories: \'%s\'' % '\', \''.join(_lib_dirs))

_FI = _register_api(_FI, _API)

if sys.platform == 'win32':
    _functype = ctypes.WINFUNCTYPE
else:
    _functype = ctypes.CFUNCTYPE

@_functype(None, ctypes.c_int, ctypes.c_char_p)
def _error_handler(fif, message):
    raise RuntimeError('mahotas.freeimage: FreeImage error: %s' % message)

_FI.FreeImage_SetOutputMessage(_error_handler)

class FI_TYPES(object):
    FIT_UNKNOWN = 0
    FIT_BITMAP = 1
    FIT_UINT16 = 2
    FIT_INT16 = 3
    FIT_UINT32 = 4
    FIT_INT32 = 5
    FIT_FLOAT = 6
    FIT_DOUBLE = 7
    FIT_COMPLEX = 8
    FIT_RGB16 = 9
    FIT_RGBA16 = 10
    FIT_RGBF = 11
    FIT_RGBAF = 12

    dtypes = {
        FIT_BITMAP: np.uint8,
        FIT_UINT16: np.uint16,
        FIT_INT16: np.int16,
        FIT_UINT32: np.uint32,
        FIT_INT32: np.int32,
        FIT_FLOAT: np.float32,
        FIT_DOUBLE: np.float64,
        FIT_COMPLEX: np.complex128,
        FIT_RGB16: np.uint16,
        FIT_RGBA16: np.uint16,
        FIT_RGBF: np.float32,
        FIT_RGBAF: np.float32
        }

    fi_types = {
        (np.uint8, 1): FIT_BITMAP,
        (np.uint8, 3): FIT_BITMAP,
        (np.uint8, 4): FIT_BITMAP,
        (np.uint16, 1): FIT_UINT16,
        (np.int16, 1): FIT_INT16,
        (np.uint32, 1): FIT_UINT32,
        (np.int32, 1): FIT_INT32,
        (np.float32, 1): FIT_FLOAT,
        (np.float64, 1): FIT_DOUBLE,
        (np.complex128, 1): FIT_COMPLEX,
        (np.uint16, 3): FIT_RGB16,
        (np.uint16, 4): FIT_RGBA16,
        (np.float32, 3): FIT_RGBF,
        (np.float32, 4): FIT_RGBAF
        }

    extra_dims = {
        FIT_UINT16: [],
        FIT_INT16: [],
        FIT_UINT32: [],
        FIT_INT32: [],
        FIT_FLOAT: [],
        FIT_DOUBLE: [],
        FIT_COMPLEX: [],
        FIT_RGB16: [3],
        FIT_RGBA16: [4],
        FIT_RGBF: [3],
        FIT_RGBAF: [4]
        }

    @classmethod
    def get_type_and_shape(cls, bitmap):
        w = _FI.FreeImage_GetWidth(bitmap)
        h = _FI.FreeImage_GetHeight(bitmap)
        fi_type = _FI.FreeImage_GetImageType(bitmap)
        if not fi_type:
            raise ValueError('mahotas.freeimage: unknown image pixel type')
        dtype = cls.dtypes[fi_type]
        if fi_type == cls.FIT_BITMAP:
            bpp = _FI.FreeImage_GetBPP(bitmap)
            if bpp == 1:
                # This is a special case
                return 'bit', None
            elif bpp == 8:
                extra_dims = []
            elif bpp == 16:
                extra_dims = []
                dtype = np.uint16
            elif bpp == 24:
                extra_dims = [3]
            elif bpp == 32:
                extra_dims = [4]
            else:
                raise ValueError('mahotas.freeimage: cannot convert %d BPP bitmap' % bpp)
        else:
            extra_dims = cls.extra_dims[fi_type]
        return np.dtype(dtype), extra_dims + [w, h]

class IO_FLAGS(object):
    #Bmp
    BMP_DEFAULT = 0
    BMP_SAVE_RLE = 1

    #Png
    PNG_DEFAULT = 0
    PNG_IGNOREGAMMA = 1

    #Gif
    GIF_DEFAULT = 0
    GIF_LOAD256 = 1
    GIF_PLAYBACK = 2

    #Ico
    ICO_DEFAULT = 0
    ICO_MAKEALPHA = 1

    #Tiff
    TIFF_DEFAULT = 0
    TIFF_CMYK = 0x0001
    TIFF_NONE = 0x0800
    TIFF_PACKBITS = 0x0100
    TIFF_DEFLATE = 0x0200
    TIFF_ADOBE_DEFLATE = 0x0400
    TIFF_CCITTFAX3 = 0x1000
    TIFF_CCITTFAX4 = 0x2000
    TIFF_LZW = 0x4000
    TIFF_JPEG = 0x8000

    #Jpeg
    JPEG_DEFAULT = 0
    JPEG_FAST = 1
    JPEG_ACCURATE = 2
    JPEG_QUALITYSUPERB = 0x80
    JPEG_QUALITYGOOD = 0x100
    JPEG_QUALITYNORMAL = 0x200
    JPEG_QUALITYAVERAGE = 0x400
    JPEG_QUALITYBAD = 0x800
    JPEG_CMYK = 0x1000
    JPEG_PROGRESSIVE = 0x2000

    #Others...
    CUT_DEFAULT = 0
    DDS_DEFAULT = 0
    HDR_DEFAULT = 0
    IFF_DEFAULT = 0
    KOALA_DEFAULT = 0
    LBM_DEFAULT = 0
    MNG_DEFAULT = 0
    PCD_DEFAULT = 0
    PCD_BASE = 1
    PCD_BASEDIV4 = 2
    PCD_BASEDIV16 = 3
    PCX_DEFAULT = 0
    PNM_DEFAULT = 0
    PNM_SAVE_RAW = 0
    PNM_SAVE_ASCII = 1
    PSD_DEFAULT = 0
    RAS_DEFAULT = 0
    TARGA_DEFAULT = 0
    TARGA_LOAD_RGB888 = 1
    WBMP_DEFAULT = 0
    XBM_DEFAULT = 0

class METADATA_MODELS(object):
    FIMD_NODATA = -1
    FIMD_COMMENTS = 0
    FIMD_EXIF_MAIN = 1
    FIMD_EXIF_EXIF = 2
    FIMD_EXIF_GPS = 3
    FIMD_EXIF_MAKERNOTE = 4
    FIMD_EXIF_INTEROP = 5
    FIMD_IPTC = 6
    FIMD_XMP = 7
    FIMD_GEOTIFF = 8
    FIMD_ANIMATION = 9
    FIMD_CUSTOM = 10

class FI_FORMAT(object):
    FIF_UNKNOWN     = -1
    FIF_BMP         =  0
    FIF_ICO         =  1
    FIF_JPEG        =  2
    FIF_JNG         =  3
    FIF_KOALA       =  4
    FIF_LBM         =  5
    FIF_IFF         = FIF_LBM
    FIF_MNG         =  6
    FIF_PBM         =  7
    FIF_PBMRAW      =  8
    FIF_PCD         =  9
    FIF_PCX         = 10
    FIF_PGM         = 11
    FIF_PGMRAW      = 12
    FIF_PNG         = 13
    FIF_PPM         = 14
    FIF_PPMRAW      = 15
    FIF_RAS         = 16
    FIF_TARGA       = 17
    FIF_TIFF        = 18
    FIF_WBMP        = 19
    FIF_PSD         = 20
    FIF_CUT         = 21
    FIF_XBM         = 22
    FIF_XPM         = 23
    FIF_DDS         = 24
    FIF_GIF         = 25
    FIF_HDR         = 26
    FIF_FAXG3       = 27
    FIF_SGI         = 28
    FIF_EXR         = 29
    FIF_J2K         = 30
    FIF_JP2         = 31
    FIF_PFM         = 32
    FIF_PICT        = 33
    FIF_RAW         = 34

def read(filename, flags=0):
    """Read an image to a numpy array of shape (width, height) for
    greyscale images, or shape (width, height, nchannels) for RGB or
    RGBA images.

    """
    bitmap = _read_bitmap(filename, flags)
    try:
        return _array_from_bitmap(bitmap)
    finally:
        _FI.FreeImage_Unload(bitmap)

def read_multipage(filename, flags=0):
    """Read a multipage image to a list of numpy arrays, where each
    array is of shape (width, height) for greyscale images, or shape
    (nchannels, width, height) for RGB or RGBA images.

    """
    ftype = _FI.FreeImage_GetFileType(filename, 0)
    if ftype == -1:
        raise ValueError('mahotas.freeimage: cannot determine type of file %s'%filename)
    create_new = False
    read_only = True
    keep_cache_in_memory = True
    multibitmap = _FI.FreeImage_OpenMultiBitmap(ftype, filename, create_new,
                                                read_only, keep_cache_in_memory,
                                                flags)
    if not multibitmap:
        raise ValueError('mahotas.freeimage: could not open %s as multi-page image.'%filename)
    try:
        pages = _FI.FreeImage_GetPageCount(multibitmap)
        arrays = []
        for i in range(pages):
            bitmap = _FI.FreeImage_LockPage(multibitmap, i)
            try:
                arrays.append(_array_from_bitmap(bitmap))
            finally:
                _FI.FreeImage_UnlockPage(multibitmap, bitmap, False)
        return arrays
    finally:
        _FI.FreeImage_CloseMultiBitmap(multibitmap, 0)

def _read_bitmap(filename, flags):
    """Load a file to a FreeImage bitmap pointer"""
    ftype = _FI.FreeImage_GetFileType(str(filename), 0)
    if ftype == -1:
        raise ValueError('mahotas.freeimage: cannot determine type of file %s'%filename)
    bitmap = _FI.FreeImage_Load(ftype, filename, flags)
    if not bitmap:
        raise ValueError('mahotas.freeimage: could not load file %s'%filename)
    return bitmap

def _wrap_bitmap_bits_in_array(bitmap, shape, dtype):
    """Return an ndarray view on the data in a FreeImage bitmap. Only
    valid for as long as the bitmap is loaded (if single page) / locked
    in memory (if multipage).

    """
    pitch = _FI.FreeImage_GetPitch(bitmap)
    itemsize = dtype.itemsize

    if len(shape) == 3:
        strides = (itemsize, shape[0]*itemsize, pitch)
    else:
        strides = (itemsize, pitch)
    bits = _FI.FreeImage_GetBits(bitmap)

    class DummyArray:
        __array_interface__ = {
            'data': (bits, False),
            'strides': strides,
            'typestr': dtype.str,
            'shape': tuple(shape),
            'version' : 3,
            }

    return np.array(DummyArray(), copy=False)

def _array_from_bitmap(bitmap):
    """Convert a FreeImage bitmap pointer to a numpy array

    """
    dtype, shape = FI_TYPES.get_type_and_shape(bitmap)
    if type(dtype) == str and dtype == 'bit':
        bitmap8 = _FI.FreeImage_ConvertToGreyscale(bitmap)
        try:
            return _array_from_bitmap(bitmap8).astype(np.bool)
        finally:
            _FI.FreeImage_Unload(bitmap8)
    array = _wrap_bitmap_bits_in_array(bitmap, shape, dtype)
    # swizzle the color components and flip the scanlines to go from
    # FreeImage's BGR[A] and upside-down internal memory format to something
    # more normal
    def n(arr):
        return arr[..., ::-1].T
    if len(shape) == 3 and _FI.FreeImage_IsLittleEndian() and \
       dtype.type == np.uint8:
        b = n(array[0])
        g = n(array[1])
        r = n(array[2])
        if shape[0] == 3:
            return np.dstack( (r,g,b) )
        elif shape[0] == 4:
            a = n(array[3])
            return np.dstack( (r,g,b,a) )
        else:
            raise ValueError('mahotas.freeimage: cannot handle images of this shape (%s)' % shape)

    # We need to copy because array does *not* own its memory
    # after bitmap is freed.
    return n(array).copy()

def string_tag(bitmap, key, model=METADATA_MODELS.FIMD_EXIF_MAIN):
    """Retrieve the value of a metadata tag with the given string key as a
    string."""
    tag = ctypes.c_int()
    if not _FI.FreeImage_GetMetadata(model, bitmap, str(key),
                                     ctypes.byref(tag)):
        return
    char_ptr = ctypes.c_char * _FI.FreeImage_GetTagLength(tag)
    return char_ptr.from_address(_FI.FreeImage_GetTagValue(tag)).raw()

def write(array, filename, flags=0):
    """Write a (width, height) or (width, height, nchannels) array to
    a greyscale, RGB, or RGBA image, with file type deduced from the
    filename.

    """
    filename = str(filename)
    ftype = _FI.FreeImage_GetFIFFromFilename(filename)
    if ftype == -1:
        raise ValueError('mahotas.freeimage: cannot determine type for %s'%filename)
    bitmap, fi_type = _array_to_bitmap(array)
    try:
        if fi_type == FI_TYPES.FIT_BITMAP:
            can_write = _FI.FreeImage_FIFSupportsExportBPP(ftype,
                                      _FI.FreeImage_GetBPP(bitmap))
        else:
            can_write = _FI.FreeImage_FIFSupportsExportType(ftype, fi_type)
        if not can_write:
            raise TypeError('mahotas.freeimage: cannot save image of this format '
                            'to this file type')
        res = _FI.FreeImage_Save(ftype, bitmap, filename, flags)
        if not res:
            raise RuntimeError('mahotas.freeimage: could not save image properly.')
    finally:
        _FI.FreeImage_Unload(bitmap)

def write_multipage(arrays, filename, flags=0):
    """Write a list of (width, height) or (nchannels, width, height)
    arrays to a multipage greyscale, RGB, or RGBA image, with file type
    deduced from the filename.

    """
    ftype = _FI.FreeImage_GetFIFFromFilename(filename)
    if ftype == -1:
        raise ValueError('mahotas.freeimage: cannot determine type of file %s'%filename)
    create_new = True
    read_only = False
    keep_cache_in_memory = True
    multibitmap = _FI.FreeImage_OpenMultiBitmap(ftype, filename, create_new,
                                                read_only, keep_cache_in_memory,
                                                0)
    if not multibitmap:
        raise ValueError('mahotas.freeimage: could not open %s for writing multi-page image.' %
                         filename)
    try:
        for array in arrays:
            bitmap,_ = _array_to_bitmap(array)
            _FI.FreeImage_AppendPage(multibitmap, bitmap)
    finally:
        _FI.FreeImage_CloseMultiBitmap(multibitmap, flags)

def _array_to_bitmap(array):
    """Allocate a FreeImage bitmap and copy a numpy array into it.

    """
    shape = array.shape
    dtype = array.dtype
    r,c = shape[:2]
    if len(shape) == 2:
        n_channels = 1
        w_shape = (c,r)
    elif len(shape) == 3:
        n_channels = shape[2]
        w_shape = (n_channels,c,r)
    else:
        raise ValueError('mahotas.freeimage: cannot handle image of 4 dimensions')
    try:
        fi_type = FI_TYPES.fi_types[(dtype.type, n_channels)]
    except KeyError:
        raise ValueError('mahotas.freeimage: cannot write arrays of given type and shape.')

    itemsize = array.dtype.itemsize
    bpp = 8 * itemsize * n_channels
    bitmap = _FI.FreeImage_AllocateT(fi_type, c, r, bpp, 0, 0, 0)
    if not bitmap:
        raise RuntimeError('mahotas.freeimage: could not allocate image for storage')
    try:
        def n(arr): # normalise to freeimage's in-memory format
            return arr.T[:,::-1]
        wrapped_array = _wrap_bitmap_bits_in_array(bitmap, w_shape, dtype)
        # swizzle the color components and flip the scanlines to go to
        # FreeImage's BGR[A] and upside-down internal memory format
        if len(shape) == 3 and _FI.FreeImage_IsLittleEndian() and \
               dtype.type == np.uint8:
            wrapped_array[0] = n(array[:,:,2])
            wrapped_array[1] = n(array[:,:,1])
            wrapped_array[2] = n(array[:,:,0])
            if shape[2] == 4:
                wrapped_array[3] = n(array[:,:,3])
        else:
            wrapped_array[:] = n(array)

        return bitmap, fi_type
    except:
      _FI.FreeImage_Unload(bitmap)
      raise


def imsavetoblob(img, filetype, flags=0):
    '''
    s = imsavetoblob(img, filetype, flags=0)

    Save `img` to a `str` object

    Parameters
    ----------
    img : ndarray
        input image
    filetype : str or integer
        A file name like string, used only to determine the file type.
        Alternatively, an integer flag (from FI_FORMAT).
    flags : integer, optional

    Returns
    -------
    s : str
        byte representation of `img` in format `filetype`
    '''
    if type(filetype) == str:
        ftype = _FI.FreeImage_GetFIFFromFilename(filetype)
    else:
        ftype = filetype
    try:
        bitmap, fi_type = _array_to_bitmap(img)
        mem = _FI.FreeImage_OpenMemory(0,0)
        if not _FI.FreeImage_SaveToMemory(ftype, bitmap, mem, flags):
            raise IOError('mahotas.freeimage.imsavetoblob: Cannot save to memory.')
        data = ctypes.c_void_p()
        size = ctypes.c_int()
        _FI.FreeImage_AcquireMemory(mem, ctypes.byref(data), ctypes.byref(size))
        return ctypes.string_at(data, size)
    finally:
        _FI.FreeImage_CloseMemory(mem)


def imreadfromblob(blob, ftype=None, as_grey=False):
    '''
    arr = imreadfromblob(blob, ftype={auto}, as_grey=False)

    Read an image from a blob (string)

    Parameters
    ----------
    blob : str
        Input
    filetype : integer, optional
        input type. By default, infer from image.
    as_grey : boolean, optional
        whether to convert colour images to grey scale

    Returns
    -------
    arr : ndarray
    '''
    try:
        mem = _FI.FreeImage_OpenMemory(blob, len(blob))
        if ftype is None:
            ftype = _FI.FreeImage_GetFileTypeFromMemory(mem, 0)
        bitmap = _FI.FreeImage_LoadFromMemory(ftype, mem, 0)
        img = _array_from_bitmap(bitmap)
        if as_grey and len(img.shape) == 3:
            # these are the values that wikipedia says are typical
            transform = np.array([ 0.30,  0.59,  0.11])
            return np.dot(img, transform)
        return img
    finally:
        _FI.FreeImage_CloseMemory(mem)


def imread(filename, as_grey=False):
    """
    img = imread(filename, as_grey=False)

    Reads an image from file `filename`

    Parameters
    ----------
      filename : file name
      as_grey : Whether to convert to grey scale image (default: no)

    Returns
    -------
      img : ndarray
    """
    img = read(filename)
    if as_grey and len(img.shape) == 3:
        # these are the values that wikipedia says are typical
        transform = np.array([ 0.30,  0.59,  0.11])
        return np.dot(img, transform)
    return img

def imsave(filename, img):
    '''
    imsave(filename, img)

    Save image to disk

    Image type is inferred from filename

    Parameters
    ----------
      filename : file name
      img : image to be saved as nd array
    '''
    write(img, filename)
