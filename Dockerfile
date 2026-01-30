FROM python:3.11-slim

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


COPY . /content
WORKDIR /content
ENV PLANZERO_DATA="/content/data"
CMD ["fastapi", "run"]
