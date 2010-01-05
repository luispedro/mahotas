_morph.so: _morph.cpp numpypp/array.hpp numpypp/dispatch.hpp
	cd .. && python setup.py build_ext --build-lib=. -U NDEBUG
	mv ../_morph.so .
