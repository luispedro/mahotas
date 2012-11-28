all: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	DEBUG=2 python setup.py build --build-lib=.

clean:
	rm -rf build mahotas/*.so mahotas/features/*.so

tests: all
	nosetests -vx

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	echo python setup.py upload_docs

.PHONY: clean docs tests all

