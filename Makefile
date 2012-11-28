all: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	DEBUG=2 python setup.py build --build-lib=.

clean:
	rm -rf build mahotas/*.so mahotas/features/*.so

tests: all
	nosetests -vx
