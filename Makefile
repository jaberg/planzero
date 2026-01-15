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
		pytest --maxfail=2 .

test_mapml: .build
	docker run \
		-v ${PWD}:/mnt/ \
		-w /mnt/ \
		-it --rm $(target) \
		pytest --maxfail=2 -vv planzero/test_mapml.py


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
