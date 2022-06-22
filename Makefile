all: standalone_all q2_all
standalone_all: standalone_stylecheck standalone_test
q2_all: q2_test q2_stylecheck
test_all: standalone_test q2_test
stylecheck_all: standalone_stylecheck q2_stylecheck

standalone_stylecheck:
	flake8 qupid/*.py
	flake8 qupid/cli/*.py
	flake8 qupid/tests/*.py
	flake8 setup.py

q2_stylecheck:
	flake8 qupid/q2/*.py

standalone_test:
	pytest --cov-report term-missing --cov=qupid --cov-config=config/standalone_cov.ini qupid/tests/ --cov-branch

q2_test:
	pytest --cov-report term-missing --cov=./qupid/q2 --cov-config=config/q2_cov.ini qupid/q2/tests/ --cov-branch
