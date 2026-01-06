FROM python:3.11-slim

#RUN apt-get update -y
#RUN apt-get install -y build-essential

WORKDIR /app

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create and activate virtualenv
RUN virtualenv venv

# Set the virtualenv as the default Python
ENV PATH="/app/venv/bin:$PATH"

# Install packages in the virtualenv
RUN pip install --no-cache-dir numpy
#RUN pip install --no-cache-dir pymc jupyter
#RUN pip install --no-cache-dir plotly
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir pint
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir networkx
RUN pip install --no-cache-dir fastapi[standard]
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir pytest
RUN pip install --no-cache-dir jupyter


COPY . /content
WORKDIR /content
ENV PLANZERO_DATA="/content/data"
CMD ["fastapi", "run"]
