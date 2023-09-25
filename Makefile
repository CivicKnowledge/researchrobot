

publish:
	git push --tags origin
	python setup.py sdist
	twine upload dist/*

develop:
	python setup.py develop

test:
	cd tests &&  python -m pytest
