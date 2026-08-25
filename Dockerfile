FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir virtualenv
RUN virtualenv venv
ENV PATH="/app/venv/bin:$PATH"
COPY ./base_requirements.txt base_requirements.txt
RUN pip install --no-cache-dir -r base_requirements.txt

FROM base AS testing
# intermediate stage, not used directly in e.g. Makefile
RUN apt-get update
RUN apt-get install -y build-essential
COPY ./requirements_dev.txt requirements_dev.txt
RUN pip install --no-cache-dir -r requirements_dev.txt


FROM testing AS development

RUN apt-get install -y tmux ncurses-base
RUN apt-get install -y git git-lfs
RUN apt-get install -y curl
RUN apt-get install -y nodejs npm # for pyright neovim plugin

# TODO: arg/logic to configure architecture here:
ENV NVIM_ARCH="arm64"
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-$NVIM_ARCH.tar.gz
RUN tar -C /opt -xzf nvim-linux-$NVIM_ARCH.tar.gz

# add to bashrc so these vars are set inside tmux shells
RUN echo 'export PATH=/app/venv/bin:$PATH' >> /root/.bashrc   # virtualenv
RUN echo 'export PATH=$PATH:/opt/nvim-linux-$NVIM_ARCH/bin' >> /root/.bashrc  # neovim

# built on dev machine
# run on dev machine
# used for most dev activities in Makefile
ENV PLANZERO_DATA="/mnt/data"
ENV PLANZERO_USE_DISK_CACHE="0"
ENV PLANZERO_APP_CACHE_DIR="/mnt/.planzero_app_cache"
ENV PLANZERO_CACHE_DIR="/mnt/.planzero_cache"
ENV PLANZERO_MODEL_CACHE_ROOT="/mnt/.planzero_model_cache_root"
ENV PLANZERO_HOME_SHOW_PLANNED_POSTS=1
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=1
# TODO: pull in the source code, data etc. to run dockerized tests
# Currently, Makefile runs tests in mount directory
#CMD ["pytest"]


FROM testing AS build_cache
# built on dev machine (GH workflow requires inference results)
# run on dev machine
ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="1"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_APP_CACHE_DIR="/content/.planzero_app_cache"
ENV PLANZERO_MODEL_CACHE_ROOT="/content/.planzero_model_cache_root"
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



FROM --platform=linux/amd64 python:3.11-slim AS gh_workflow

# MAINTAIN: COPY-PASTE FROM base
WORKDIR /app
RUN pip install --no-cache-dir virtualenv
RUN virtualenv venv
ENV PATH="/app/venv/bin:$PATH"
COPY ./base_requirements.txt base_requirements.txt
RUN pip install --no-cache-dir -r base_requirements.txt

# app cache copied into CWD by .github/workflows/test.yaml artifact download
COPY _planzero_app_cache /content/_planzero_app_cache

# MAINTAIN COPY-PASTE FROM production_server
COPY ./planzero /content/planzero
COPY ./html /content/html
COPY ./app.py /content/app.py
COPY ./data/EN_GHG_IPCC_Can_Prov_Terr.csv /content/data/EN_GHG_IPCC_Can_Prov_Terr.csv
WORKDIR /content

ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="1"
ENV PLANZERO_CACHE_DIR="/content/_planzero_cache"
ENV PLANZERO_APP_CACHE_DIR="/content/_planzero_app_cache"
ENV PLANZERO_HOME_SHOW_PLANNED_POSTS=0
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=0
CMD ["fastapi", "run"]
