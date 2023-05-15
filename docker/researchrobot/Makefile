
.PHONY: build


CONFIG=$(shell ds_config -f)
RASTER=$(shell ds config -r)
PWD = $(shell pwd)

CBSA_PKG='shape+https://www2.census.gov/geo/tiger/TIGER2020/CBSA/tl_2020_us_cbsa.zip'

demosearch:
	cd ../../demosearch && \
	rm -rf dist && \
	python setup.py sdist &&\
	cp dist/`python setup.py --fullname`.tar.gz $(PWD)/demosearch.tar.gz

radsrv:
	cd ../../radsrv && \
	rm -rf dist && \
	python setup.py sdist &&\
	cp dist/`python setup.py --fullname`.tar.gz $(PWD)/radsrv.tar.gz

# Make and copy the python packages
pypkg: demosearch radsrv

config:
	cat  $(CONFIG) | sed 's/^cache_path.*/cache_path:  \/cache\/radius/' > _ds_config.yaml


build:
	docker-compose build

search-index:
	docker-compose run radsrv ds_cachedata -i $(CBSA_PKG)

# Copy the local raster file to the server
cpraster:
	docker compose cp `ds_config -r` shell:/cache/radius/

initialize:
	docker-compose start shell
	docker-compose run shell mkdir -p /cache/radius
	docker compose cp `ds_config -r` shell:/cache/radius/
	docker-compose run shell ds_cachedata -i $(CBSA_PKG)

full_build: pypkg build initialize



restart:
	docker-compose down
	docker-compose up -d

redeploy: pypkg build restart
	echo 'Done'
