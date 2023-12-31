# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

ARG GO_VERSION=1.21.5
ARG HUGO_VERSION=0.121.1
ARG ARCHITECTURE=amd64

# install git, go and hugo
RUN  apt update && apt upgrade -y && apt install -y git wget
RUN wget https://go.dev/dl/go${GO_VERSION}.linux-${ARCHITECTURE}.tar.gz && tar -C /usr/local -xzf go${GO_VERSION}.linux-${ARCHITECTURE}.tar.gz 
RUN export PATH=$PATH:/usr/local/go/bin:/usr/local && go version
RUN wget https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-${ARCHITECTURE}.tar.gz && tar -C /usr/local -xzf hugo_extended_${HUGO_VERSION}_linux-${ARCHITECTURE}.tar.gz && ls -al /usr/local
RUN /usr/local/hugo version

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /app
COPY . /app

# Use Poetry to install dependencies
RUN poetry config virtualenvs.create true && poetry install --no-interaction --no-ansi

# Run guidance-engine.py when the container launches
CMD ["poetry", "run", "python", "guidance_engine_adapter.py"]
