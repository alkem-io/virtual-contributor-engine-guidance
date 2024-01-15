# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye as builder

# Set the working directory in the container to /app
WORKDIR /app

ARG GO_VERSION=1.21.6
ARG HUGO_VERSION=0.121.2
ARG TARGETARCH

# install git, go and hugo
RUN  apt update && apt upgrade -y && apt install -y git wget
RUN wget https://go.dev/dl/go${GO_VERSION}.linux-${TARGETARCH}.tar.gz && tar -C /usr/local -xzf go${GO_VERSION}.linux-${TARGETARCH}.tar.gz 
RUN export PATH=$PATH:/usr/local/go/bin:/usr/local && go version
RUN wget https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz && tar -C /usr/local -xzf hugo_extended_${HUGO_VERSION}_linux-${TARGETARCH}.tar.gz && ls -al /usr/local
RUN /usr/local/hugo version

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /app
COPY . /app

# Use Poetry to install dependencies
RUN poetry config virtualenvs.create true && poetry install --no-interaction --no-ansi

# Start a new, final stage
FROM python:${PYTHON_VERSION}-slim-bullseye

WORKDIR /app

# Install git and Poetry in the final stage
RUN apt update && apt install -y git && pip install poetry

# Copy the compiled app, Hugo executable, Go, and the virtual environment from the previous stage
COPY --from=builder /app /app
COPY --from=builder /usr/local/hugo /usr/local/hugo
COPY --from=builder /usr/local/go /usr/local/go
COPY --from=builder /root/.cache/pypoetry/virtualenvs /root/.cache/pypoetry/virtualenvs

# Add Hugo and Go to the PATH
ENV PATH="/usr/local/hugo:/usr/local/go/bin:${PATH}"

# Run guidance_engine.py when the container launches
CMD ["poetry", "run", "python", "guidance_engine.py"]