target = ${PROJECTNAME}

.build.base: Dockerfile
	docker build --target base -t $(target):base .
	touch .build.base

.build.test: Dockerfile
	docker build --target testing -t $(target):test .
	touch .build.test

.build.prod: Dockerfile
	docker build --target production -t $(target):prod .
	touch .build.prod

bash: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		bash

jupyter: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8013:8013 \
		-w /mnt/ \
		-it --rm $(target):test \
		jupyter lab --port=8013 --ip 0.0.0.0 --no-browser --allow-root

test: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		pytest -W error --maxfail=2 .

test_200_internal: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		pytest -W error -vv -k internal test_200.py

test_200: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		pytest -W error --maxfail=1 -vv -k endpoints test_200.py

local: .build.test
	docker run \
		-e PLANZERO_DATA=/mnt/data/ \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8012:8012 \
		-w /mnt/ \
		-it --rm $(target):test \
		fastapi dev --port=8012 --host=0.0.0.0

prodlike: .build.prod
	docker run \
		-p 127.0.0.1:8015:8015 \
		-it --rm $(target):prod \
		fastapi run --port=8015 --host=0.0.0.0

deploy:
	fly deploy

clean:
	rm .build.*

cache_ghgrp_by_petrinex: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_ghgrp_by_petrinex

petrinex_unzip_download:
	# when you download using petrinex website it gives you a download.zip
	# file with whatever months you asked for, for whatever provinces you
	# asked for
	(cd data/petrinex && unzip ~/Downloads/download.zip)

petrinex_rm_download:
	# N.B this is OUTSIDE THIS FOLDER
	rm ~/Downloads/download.zip

cache_petrinex_SK_2022: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2022 --PT=Saskatchewan --large-emitter-cutoff-monthly=5

cache_petrinex_AB_2022: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2022 --PT=Alberta --large-emitter-cutoff-monthly=5

cache_petrinex_SK_2023: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2023 --PT=Saskatchewan --large-emitter-cutoff-monthly=5

cache_petrinex_AB_2023: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2023 --PT=Alberta --large-emitter-cutoff-monthly=5

cache_petrinex_SK_2024: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2024 --PT=Saskatchewan --large-emitter-cutoff-monthly=5

cache_petrinex_AB_2024: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2024 --PT=Alberta --large-emitter-cutoff-monthly=5

cache_petrinex_SK_2025: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero cache_petrinex --year=2025 --PT=Saskatchewan

request_all_planzero_pages: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero request_all_pages

warmup_cache_speed_test: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python warmup.py

demo_neud: .build.test
	# TODO: move this into __main__.py
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero.neud

demo_sc_32_10_0130_01: .build.test
	# TODO: move this into __main__.py
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero.sc_3210013001


html/blog/2026-04-03-bovaer_assets: .build.test
	# these asset files are meant to be stored in git
	# the script is used during development to re-generate them
	# after the post is beyond amendment, this build script could be removed
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -c "import planzero; planzero.blog.ModellingBovaer.generate_assets()"


cache/inference/Static_Normals/sentinel: .build.test
	# Perform inference for the Static_Normals model
	# save multiple files in this directory
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero.nir_constant_predictor


cache/inference/AR2/sentinel: .build.test
	# Perform inference for the AR2 model
	# save multiple files in this directory
	#
	# Slow to run though! Takes maybe 2 hours?
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_USE_DISK_CACHE=0 \
		-w /mnt/ \
		-it --rm $(target):test \
		python -m planzero nir_ar2_inference

html/blog/2026-05-26-probabilistic-modelling-assets: .build.test
	# these asset files are meant to be stored in git
	# the script is used during development to re-generate them
	# after the post is beyond amendment, this build script could be removed
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -c "import planzero; planzero.blog.TwoProbabilisticModels.generate_assets()"
