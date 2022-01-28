all: stylecheck test

stylecheck:
	flake8 matchlock/*.py
	flake8 tests/*.py
	flake8 setup.py

test:
	pytest tests/*.py
