project(
    'mahotas',
    'cpp',
    version: run_command(
        find_program('python', 'python3'),
        files('build-aux/meson-helper.py'),
        'version',
        check: true,
    ).stdout().strip(),
    license: 'MIT',
    meson_version: '>=1.2.3',
)

py = import('python').find_installation(pure: false)
helper = files('build-aux/meson-helper.py')

numpy_include_dir = run_command(
    py,
    helper,
    'numpy-include',
    check: true,
).stdout().strip()

cpp = meson.get_compiler('cpp')
if cpp.get_argument_syntax() == 'msvc'
    common_cpp_args = [
        '/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '/DPY_ARRAY_UNIQUE_SYMBOL=Mahotas_PyArray_API_Symbol',
        '/EHsc',
    ]
    if get_option('glibcpp_debug')
        common_cpp_args += ['/D_GLIBCXX_DEBUG=1']
    endif
else
    common_cpp_args = [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-DPY_ARRAY_UNIQUE_SYMBOL=Mahotas_PyArray_API_Symbol',
    ]
    if get_option('glibcpp_debug')
        common_cpp_args += ['-D_GLIBCXX_DEBUG=1']
    endif
endif

python_sources = files(
    'mahotas/__init__.py',
    'mahotas/_filters.py',
    'mahotas/bbox.py',
    'mahotas/bwperim.py',
    'mahotas/center_of_mass.py',
    'mahotas/colors.py',
    'mahotas/convolve.py',
    'mahotas/demos/__init__.py',
    'mahotas/demos/distance.py',
    'mahotas/demos/edge_demo.py',
    'mahotas/demos/morphology.py',
    'mahotas/demos/nuclear.py',
    'mahotas/demos/nuclear_distance_watershed.py',
    'mahotas/demos/superpixels.py',
    'mahotas/demos/surf_gaussians.py',
    'mahotas/demos/surf_luispedro.py',
    'mahotas/demos/thresholding.py',
    'mahotas/demos/wally.py',
    'mahotas/demos/wavelet_compress.py',
    'mahotas/distance.py',
    'mahotas/edge.py',
    'mahotas/euler.py',
    'mahotas/features/__init__.py',
    'mahotas/features/lbp.py',
    'mahotas/features/moments.py',
    'mahotas/features/shape.py',
    'mahotas/features/surf.py',
    'mahotas/features/tas.py',
    'mahotas/features/texture.py',
    'mahotas/features/zernike.py',
    'mahotas/features_cli.py',
    'mahotas/freeimage.py',
    'mahotas/histogram.py',
    'mahotas/internal.py',
    'mahotas/interpolate.py',
    'mahotas/io/__init__.py',
    'mahotas/io/freeimage.py',
    'mahotas/io/pil.py',
    'mahotas/labeled.py',
    'mahotas/lbp.py',
    'mahotas/mahotas_version.py',
    'mahotas/moments.py',
    'mahotas/morph.py',
    'mahotas/polygon.py',
    'mahotas/resize.py',
    'mahotas/segmentation.py',
    'mahotas/stretch.py',
    'mahotas/surf.py',
    'mahotas/tas.py',
    'mahotas/tests/__init__.py',
    'mahotas/tests/pymorph_copy.py',
    'mahotas/tests/test_bbox.py',
    'mahotas/tests/test_bwperim.py',
    'mahotas/tests/test_center_of_mass.py',
    'mahotas/tests/test_citation.py',
    'mahotas/tests/test_close_holes.py',
    'mahotas/tests/test_colors.py',
    'mahotas/tests/test_convolve.py',
    'mahotas/tests/test_demos.py',
    'mahotas/tests/test_dilate_erode.py',
    'mahotas/tests/test_distance.py',
    'mahotas/tests/test_edge.py',
    'mahotas/tests/test_euler.py',
    'mahotas/tests/test_features_shape.py',
    'mahotas/tests/test_filters.py',
    'mahotas/tests/test_find.py',
    'mahotas/tests/test_freeimage.py',
    'mahotas/tests/test_gvoronoi.py',
    'mahotas/tests/test_histogram.py',
    'mahotas/tests/test_hitmiss.py',
    'mahotas/tests/test_imresize.py',
    'mahotas/tests/test_internal.py',
    'mahotas/tests/test_interpolate.py',
    'mahotas/tests/test_io.py',
    'mahotas/tests/test_label.py',
    'mahotas/tests/test_labeled.py',
    'mahotas/tests/test_lbp.py',
    'mahotas/tests/test_mahotas.py',
    'mahotas/tests/test_majority.py',
    'mahotas/tests/test_mean_filter.py',
    'mahotas/tests/test_median_filter.py',
    'mahotas/tests/test_moments.py',
    'mahotas/tests/test_morph.py',
    'mahotas/tests/test_polygon.py',
    'mahotas/tests/test_segmentation.py',
    'mahotas/tests/test_stretch.py',
    'mahotas/tests/test_surf.py',
    'mahotas/tests/test_surf_regression.py',
    'mahotas/tests/test_tas.py',
    'mahotas/tests/test_template_match.py',
    'mahotas/tests/test_texture.py',
    'mahotas/tests/test_thin.py',
    'mahotas/tests/test_thresholding.py',
    'mahotas/tests/test_watershed.py',
    'mahotas/tests/test_zernike.py',
    'mahotas/tests/utils.py',
    'mahotas/texture.py',
    'mahotas/thin.py',
    'mahotas/thresholding.py',
    'mahotas/zernike.py',
)

package_data = files(
    'mahotas/demos/data/DepartmentStore.jpg',
    'mahotas/demos/data/lena.jpg',
    'mahotas/demos/data/luispedro.jpg',
    'mahotas/demos/data/nuclear.png',
    'mahotas/tests/data/1bpp.bmp',
    'mahotas/tests/data/determinant_zero.png',
    'mahotas/tests/data/luispedro.npy',
    'mahotas/tests/data/rgba.png',
)

py.install_sources(
    python_sources,
    pure: false,
    preserve_path: true,
)

install_data(
    package_data,
    install_dir: py.get_install_dir(),
    preserve_path: true,
)

module_include_dirs = include_directories(numpy_include_dir)

shared_extension_sources = {
    '_bbox': ['mahotas/_bbox.cpp'],
    '_center_of_mass': ['mahotas/_center_of_mass.cpp'],
    '_convex': ['mahotas/_convex.cpp'],
    '_convolve': ['mahotas/_convolve.cpp', 'mahotas/_filters.cpp'],
    '_distance': ['mahotas/_distance.cpp'],
    '_histogram': ['mahotas/_histogram.cpp'],
    '_interpolate': ['mahotas/_interpolate.cpp', 'mahotas/_filters.cpp'],
    '_labeled': ['mahotas/_labeled.cpp', 'mahotas/_filters.cpp'],
    '_morph': ['mahotas/_morph.cpp', 'mahotas/_filters.cpp'],
    '_thin': ['mahotas/_thin.cpp'],
}

foreach name, sources : shared_extension_sources
    py.extension_module(
        name,
        sources,
        cpp_args: common_cpp_args,
        include_directories: module_include_dirs,
        install: true,
        subdir: 'mahotas',
    )
endforeach

feature_extension_sources = {
    '_lbp': ['mahotas/features/_lbp.cpp'],
    '_surf': ['mahotas/features/_surf.cpp'],
    '_texture': ['mahotas/features/_texture.cpp', 'mahotas/_filters.cpp'],
    '_zernike': ['mahotas/features/_zernike.cpp'],
}

foreach name, sources : feature_extension_sources
    py.extension_module(
        name,
        sources,
        cpp_args: common_cpp_args,
        include_directories: module_include_dirs,
        install: true,
        subdir: 'mahotas/features',
    )
endforeach
