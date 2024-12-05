# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye AS builder

# Set the working directory in the container to /app
WORKDIR /app

ARG GO_VERSION=1.21.6
ARG HUGO_VERSION=0.121.2
ARG TARGETARCH

# install git, go and hugo
RUN apt-get upgrade -y 
RUN apt-get update -y 
RUN apt-get install -y git wget
RUN wget https://go.dev/dl/go${GO_VERSION}.linux-${TARGETARCH}.tar.gz && tar -C /usr/local -xzf go${GO_VERSION}.linux-${TARGETARCH}.tar.gz 

RUN export PATH=$PATH:/usr/local/go/bin:/usr/local && go version

RUN wget https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz && tar -C /bin -xzf hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz && ls -al /usr/local
RUN hugo version

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /app
COPY . /app

# # Use Poetry to install dependencies
RUN poetry config virtualenvs.create true && poetry install --no-interaction --no-ansi

# Run main.py when the container launches
CMD ["poetry", "run", "python", "main.py"]
