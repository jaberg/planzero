target = ${PROJECTNAME}

.build.base: Dockerfile
	docker build --target base -t $(target):base .
	touch .build.base

.build.dev: Dockerfile
	docker build --target testing -t $(target):test .
	# touch .build.test   # uncomment to skip build command

.build.dev: Dockerfile
	docker build --target development -t $(target):dev .
	# touch .build.test   # uncomment to skip build command


.build.cache: Dockerfile
	docker build --target build_cache -t $(target):cache .
	touch .build.cache


.build.prod: Dockerfile
	docker build --target production_server -t $(target):prod .
	touch .build.prod


tmux: .build.dev
	docker run \
		-v ${PWD}:/mnt/ \
		-v ~/.config/git:/root/.config/git \
		-v ~/.config/nvim:/root/.config/nvim \
		-v ~/.config/tmux:/root/.config/tmux \
		-v ~/.ssh:/root/.ssh \
		-e TERM=xterm-256color \
		-e COLORTERM=truecolor \
		-w /mnt/ \
		-it --rm $(target):dev \
		tmux

bash_prod: .build.prod
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):prod \
		bash

jupyter: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8013:8013 \
		-w /mnt/ \
		-it --rm $(target):dev \
		jupyter lab --port=8013 --ip 0.0.0.0 --no-browser --allow-root

test: .build.test
	docker run \
		-v ${PWD}:/mnt/ \
		-e PLANZERO_DATA=/mnt/data \
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

local: .build.dev
	docker run \
		-e PLANZERO_DATA=/mnt/data/ \
		-v ${PWD}:/mnt/ \
		-p 127.0.0.1:8012:8012 \
		-w /mnt/ \
		-it --rm $(target):dev \
		fastapi dev --port=8012 --host=0.0.0.0

prodlike: .build.prod
	docker run \
		-p 127.0.0.1:8015:8015 \
		-it --rm $(target):prod \
		fastapi run --port=8015 --host=0.0.0.0

deploy:
	fly deploy --local-only

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


html/blog/2026-05-26-probabilistic-modelling-assets: .build.test
	# these asset files are meant to be stored in git
	# the script is used during development to re-generate them
	# after the post is beyond amendment, this build script could be removed
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target):test \
		python -c "import planzero; planzero.blog.TwoProbabilisticModels.generate_assets()"

build_and_test:
	# replicates the logic of .github/workflows/test.yaml build_and_test
	# meant to be run *inside* docker development env
	# the environment and docker command should configure the project root
	# as /mnt
	# and set the environment variables to use subfolders as cache directories
	rm -f ./my_database.db
	python -m planzero.model_db init
	python -m planzero inference_prep --model=Static_Normals_2024_12_31
	python -m planzero inference_work --model=Static_Normals_2024_12_31
	pytest -W error --maxfail=10 .
