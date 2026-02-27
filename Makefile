target = ${PROJECTNAME}

.build: Dockerfile
	docker build -t $(target) .
	touch .build

bash: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		bash

jupyter: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8013:8013 \
		-w /mnt/ \
		-it --rm $(target) \
		jupyter lab --port=8013 --ip 0.0.0.0 --no-browser --allow-root

test: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 .

test_blogs: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv -k test_each_blog test_200.py

test_200: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv test_200.py

test_ipcc_canada: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv test_ipcc_canada.py

test_mapml: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv planzero/test_mapml.py

test_co2e: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv planzero/test_co2e.py

test_est_nir: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv planzero/test_est_nir.py

test_sts: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv planzero/test_sts.py

test_objtensor: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv planzero/test_objtensor.py

test_html: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest --maxfail=2 -vv planzero/test_html.py

test_ipcc_home: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest --maxfail=2 -vv planzero/test_ipcc_home.py

test_sc_2510003001: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest --maxfail=2 -vv planzero/test_sc_2510003001.py

test_sc_25_10_0084_01: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest --maxfail=2 -vv planzero/test_sc_25_10_0084_01.py


stakeholders: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python stakeholders.py

local: .build
	docker run \
		-e PLANZERO_DATA=/mnt/data/ \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8012:8012 \
		-w /mnt/ \
		-it --rm $(target) \
		fastapi dev --port=8012 --host=0.0.0.0

prodlike: .build
	docker run \
		-p 127.0.0.1:8012:8012 \
		-it --rm $(target) \
		fastapi run --port=8012 --host=0.0.0.0

deploy:
	fly deploy

clean:
	rm .build

print_sectoral_emissions_gaps: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero
