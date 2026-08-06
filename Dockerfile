FROM python:3.11-slim AS base

#ARG GIT_COMMIT_COUNT
#ARG GIT_HEAD_HASH
#ARG R2_ACCOUNT_ID
#ARG R2_ACCESS_KEY_ID
#ARG R2_SECRET_ACCESS_KEY
#ARG R2_BUCKET_NAME

WORKDIR /app

RUN pip install --no-cache-dir virtualenv
RUN virtualenv venv
ENV PATH="/app/venv/bin:$PATH"
COPY ./base_requirements.txt base_requirements.txt
RUN pip install --no-cache-dir -r base_requirements.txt

FROM base AS development
# intermediate stage, not used directly in e.g. Makefile
RUN apt-get update
RUN apt-get install -y build-essential
RUN pip install --no-cache-dir pytest
RUN pip install --no-cache-dir jupyter
RUN pip install --no-cache-dir openpyxl xlrd
RUN pip install --no-cache-dir xarray netCDF4 matplotlib cartopy
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir jax
RUN pip install --no-cache-dir numpyro
RUN pip install --no-cache-dir boto3


FROM development AS testing
# built on dev machine
# run on dev machine
# used for most dev activities in Makefile
ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="0"
ENV PLANZERO_APP_CACHE_DIR="/content/.planzero_app_cache"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_HOME_SHOW_PLANNED_POSTS=1
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=1
# TODO: pull in the source code, data etc. to run dockerized tests
# Currently, Makefile runs tests in mount directory
#CMD ["pytest"]


FROM development AS build_cache
# built on dev machine (GH workflow requires inference results)
# run on dev machine
ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="1"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_APP_CACHE_DIR="/content/.planzero_app_cache"
ENV PLANZERO_HOME_SHOW_PLANNED_POSTS=0
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=0

# TODO: pull inference results from R2
COPY ./planzero /content/planzero
COPY ./data /content/data
COPY ./html /content/html

# TODO: GH may require additional cache elements from R2
COPY ./cache /content/cache
COPY ./warmup.py /content/warmup.py
COPY ./app.py /content/app.py
WORKDIR /content

# Build app cache
RUN python warmup.py


FROM base AS production_server
# built on dev machine
# run on dev machine or fly.io
# MAINTAIN: replicate changes in production_server_amd64
COPY --from=build_cache /content/.planzero_app_cache /content/.planzero_app_cache
COPY ./planzero /content/planzero
COPY ./html /content/html
COPY ./app.py /content/app.py
COPY ./data/EN_GHG_IPCC_Can_Prov_Terr.csv /content/data/EN_GHG_IPCC_Can_Prov_Terr.csv
WORKDIR /content

ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="1"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_APP_CACHE_DIR="/content/.planzero_app_cache"
ENV PLANZERO_HOME_SHOW_PLANNED_POSTS=0
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=0
CMD ["fastapi", "run"]


FROM --platform=linux/amd64 python:3.11-slim AS production_server_amd64

# MAINTAIN: COPY-PASTE FROM base
WORKDIR /app
RUN pip install --no-cache-dir virtualenv
RUN virtualenv venv
ENV PATH="/app/venv/bin:$PATH"
COPY ./base_requirements.txt base_requirements.txt
RUN pip install --no-cache-dir -r base_requirements.txt

# MAINTAIN COPY-PASTE FROM production_server
COPY --from=build_cache /content/.planzero_app_cache /content/.planzero_app_cache
COPY ./planzero /content/planzero
COPY ./html /content/html
COPY ./app.py /content/app.py
COPY ./data/EN_GHG_IPCC_Can_Prov_Terr.csv /content/data/EN_GHG_IPCC_Can_Prov_Terr.csv
WORKDIR /content

ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="1"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_APP_CACHE_DIR="/content/.planzero_app_cache"
ENV PLANZERO_HOME_SHOW_PLANNED_POSTS=0
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=0
CMD ["fastapi", "run"]
