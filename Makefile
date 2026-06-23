target = ${PROJECTNAME}

.build: Dockerfile
	docker build -t $(target) .
	touch .build

bash: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		bash

bash_disk_cache: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		bash

jupyter: .build
	docker run \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8013:8013 \
		-w /mnt/ \
		-it --rm $(target) \
		jupyter lab --port=8013 --ip 0.0.0.0 --no-browser --allow-root

test: .build
	# run tests with memory cache
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 .

test_blogs: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=1 -vv -k test_each_blog test_200.py

test_200_internal: .build
	# do use memory cache
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error -vv -k internal test_200.py

test_200: .build
	# do use memory cache
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=1 -vv -k endpoints test_200.py

test_ipcc_canada: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=2 -vv test_ipcc_canada.py

test_cattle: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest -W error --maxfail=1 -vv planzero/test_cattle.py

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
		-e PLANZERO_USE_DISK_CACHE=0 \
		-e PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=1 \
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


html/blog/2026-04-03-bovaer_assets:
	# these asset files are meant to be stored in git
	# the script is used during development to re-generate them
	# after the post is beyond amendment, this build script could be removed
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		python -c "import planzero; planzero.blog.ModellingBovaer.generate_assets()"

cache/inference/Static_Normals/sentinel:
	# Perform inference for the Static_Normals model
	# save multiple files in this directory
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero.nir_constant_predictor


cache/inference/AR2/sentinel:
	# Perform inference for the AR2 model
	# save multiple files in this directory
	#
	# Slow to run though! Takes maybe 2 hours?
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target) \
		python -m planzero nir_ar2_inference
