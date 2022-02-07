all: stylecheck test

stylecheck:
	flake8 qupid/*.py
	flake8 tests/*.py
	flake8 setup.py

test:
	pytest --cov-report term-missing --cov=qupid tests/
