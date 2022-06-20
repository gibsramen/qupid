all: standalone_all q2_all
standalone_all: standalone_stylecheck
q2_all: q2_stylecheck

standalone_stylecheck:
	flake8 qupid/*.py
	flake8 qupid/cli/*.py
	flake8 qupid/tests/*.py
	flake8 setup.py

q2_stylecheck:
	flake8 qupid/q2/*.py

standalone_test:
	pytest --cov-report term-missing --cov=qupid qupid/tests/ -W ignore::DeprecationWarning
