


THIS_REV=$(shell python setup.py --version)
NEXT_REV=$(shell python -c "import sys; import semantic_version; \
print( semantic_version.Version('.'.join(sys.argv[1].split('.')[:3])).next_patch()  )\
" $(THIS_REV) )

# Create a new revision
rev:
	git tag $(NEXT_REV)

showrev:
	@echo this=$(THIS_REV) next=$(NEXT_REV)

publish:
	git push --tags origin
	python setup.py sdist
	twine upload dist/*

develop:
	python setup.py develop

test:
	cd tests &&  python -m pytest
