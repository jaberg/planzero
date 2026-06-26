FROM python:3.11-slim AS base

WORKDIR /app

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create and activate virtualenv
RUN virtualenv venv

# Set the virtualenv as the default Python
ENV PATH="/app/venv/bin:$PATH"

# Install packages in the virtualenv
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir pint
RUN pip install --no-cache-dir networkx
RUN pip install --no-cache-dir fastapi[standard]
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir pytest
RUN pip install --no-cache-dir jupyter
RUN pip install --no-cache-dir markdown
RUN pip install --no-cache-dir latex2mathml
RUN pip install --no-cache-dir stats-can
RUN pip install --no-cache-dir openpyxl xlrd
RUN pip install --no-cache-dir diskcache


RUN apt-get update
RUN apt-get install -y build-essential
RUN pip install --no-cache-dir xarray netCDF4 matplotlib cartopy
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir jax
RUN pip install --no-cache-dir numpyro

COPY ./planzero /content/planzero
COPY ./data /content/data
COPY ./html /content/html
COPY ./cache /content/cache
COPY ./app.py /content/app.py
COPY ./warmup.py /content/warmup.py
WORKDIR /content


# Testing: no diskcache (memcache only), no warmup
FROM base AS testing
ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="0"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=1
CMD ["pytest"]

# Production: diskcache, warmup
FROM base AS production
ENV PLANZERO_DATA="/content/data"
ENV PLANZERO_USE_DISK_CACHE="1"
ENV PLANZERO_CACHE_DIR="/content/.planzero_cache"
ENV PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS=0
RUN python warmup.py
CMD ["fastapi", "run"]
