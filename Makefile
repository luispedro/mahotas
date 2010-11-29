all: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	python setup.py build --build-lib=.
