# Copyright (C) 2009-2025, Luis Pedro Coelho <luis@luispedro.org>

import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    import numpy
except ImportError as e:
    print(
        """
Could not import numpy ({}).

It is possible that building will fail.
        """.format(e)
    )

    class FakeNumpy:
        @staticmethod
        def get_include():
            return []

    numpy = FakeNumpy()


undef_macros = []
define_macros = []
if os.environ.get("DEBUG"):
    undef_macros = ["NDEBUG"]
    if os.environ.get("DEBUG") == "2":
        define_macros = [("_GLIBCXX_DEBUG", "1")]

define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
define_macros.append(("PY_ARRAY_UNIQUE_SYMBOL", "Mahotas_PyArray_API_Symbol"))

extensions = {
    "mahotas._bbox": ["mahotas/_bbox.cpp"],
    "mahotas._center_of_mass": ["mahotas/_center_of_mass.cpp"],
    "mahotas._convex": ["mahotas/_convex.cpp"],
    "mahotas._convolve": ["mahotas/_convolve.cpp", "mahotas/_filters.cpp"],
    "mahotas._distance": ["mahotas/_distance.cpp"],
    "mahotas._histogram": ["mahotas/_histogram.cpp"],
    "mahotas._interpolate": ["mahotas/_interpolate.cpp", "mahotas/_filters.cpp"],
    "mahotas._labeled": ["mahotas/_labeled.cpp", "mahotas/_filters.cpp"],
    "mahotas._morph": ["mahotas/_morph.cpp", "mahotas/_filters.cpp"],
    "mahotas._thin": ["mahotas/_thin.cpp"],
    "mahotas.features._lbp": ["mahotas/features/_lbp.cpp"],
    "mahotas.features._surf": ["mahotas/features/_surf.cpp"],
    "mahotas.features._texture": [
        "mahotas/features/_texture.cpp",
        "mahotas/_filters.cpp",
    ],
    "mahotas.features._zernike": ["mahotas/features/_zernike.cpp"],
}

ext_modules = [
    Extension(
        key,
        sources=sources,
        undef_macros=undef_macros,
        define_macros=define_macros,
        include_dirs=[numpy.get_include()],
    )
    for key, sources in extensions.items()
]

copt = {
    "msvc": ["/EHsc"],
    "intelw": ["/EHsc"],
}


class build_ext_subclass(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        if compiler_type in copt:
            for extension in self.extensions:
                extension.extra_compile_args = copt[compiler_type]
        super().build_extensions()


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_subclass},
)
