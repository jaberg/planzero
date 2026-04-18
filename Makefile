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

test_ghgrp: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest --maxfail=2 -vv planzero/test_ghgrp.py



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
		-p 127.0.0.1:8015:8015 \
		-it --rm $(target) \
		fastapi run --port=8015 --host=0.0.0.0

deploy:
	fly deploy

clean:
	rm .build

print_sectoral_emissions_gaps: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero print_max_gaps

cache_ghgrp_by_petrinex: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_ghgrp_by_petrinex

petrinex_unzip_download:
	# when you download using petrinex website it gives you a download.zip
	# file with whatever months you asked for, for whatever provinces you
	# asked for
	(cd data/petrinex && unzip ~/Downloads/download.zip)

petrinex_rm_download:
	rm ~/Downloads/download.zip

cache_petrinex_SK_2022: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2022 --PT=Saskatchewan --large-emitter-cutoff-monthly=5

cache_petrinex_AB_2022: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2022 --PT=Alberta --large-emitter-cutoff-monthly=5

cache_petrinex_SK_2023: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2023 --PT=Saskatchewan --large-emitter-cutoff-monthly=5

cache_petrinex_AB_2023: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2023 --PT=Alberta --large-emitter-cutoff-monthly=5

cache_petrinex_SK_2024: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2024 --PT=Saskatchewan --large-emitter-cutoff-monthly=5

cache_petrinex_AB_2024: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2024 --PT=Alberta --large-emitter-cutoff-monthly=5

cache_petrinex_SK_2025: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero cache_petrinex --year=2025 --PT=Saskatchewan

request_all_planzero_pages: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero request_all_pages

warmup_cache_speed_test: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python warmup.py

demo_neud: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero.neud

demo_sc_32_10_0130_01: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero.sc_3210013001
