debug: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	DEBUG=2 python setup.py build --build-lib=.

fast: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	python setup.py build --build-lib=.

clean:
	rm -rf build mahotas/*.so mahotas/features/*.so

tests: debug
	nosetests -vx

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	@echo python setup.py upload_docs

.PHONY: clean docs tests fast debug

