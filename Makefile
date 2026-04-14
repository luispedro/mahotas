PIP_EDITABLE = python -m pip install --editable . --no-deps --no-build-isolation

debug: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	$(PIP_EDITABLE) --config-settings=build-dir=build/debug --config-settings=setup-args=-Dbuildtype=debug --config-settings=setup-args=-Dglibcpp_debug=true

debug3: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	python3 -m pip install --editable . --no-deps --no-build-isolation --config-settings=build-dir=build/debug --config-settings=setup-args=-Dbuildtype=debug --config-settings=setup-args=-Dglibcpp_debug=true

fast: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	$(PIP_EDITABLE) --config-settings=build-dir=build/fast --config-settings=setup-args=-Dbuildtype=release

install:
	python -m pip install .

fast3: mahotas/*.cpp mahotas/*.h mahotas/*.hpp
	python3 -m pip install --editable . --no-deps --no-build-isolation --config-settings=build-dir=build/fast --config-settings=setup-args=-Dbuildtype=release

clean:
	rm -rf build dist .mesonpy-* mahotas/*.so mahotas/features/*.so

tests: debug
	pytest -v

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	@echo python -m build

.PHONY: clean docs tests fast debug install fast3 debug3
