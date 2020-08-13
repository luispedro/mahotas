debug: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	DEBUG=2 python setup.py build --build-lib=.

debug3: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	DEBUG=2 python3 setup.py build --build-lib=.

fast: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	python setup.py build --build-lib=.

install:
	python setup.py install

fast3: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	python3 setup.py build --build-lib=.

clean:
	rm -rf build mahotas/*.so mahotas/features/*.so

tests: debug
	pytest -v

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	@echo python setup.py upload_docs

.PHONY: clean docs tests fast debug install fast3 debug3

