# Use an official Python runtime as a parent image
FROM python:3-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /app
COPY . /app

# install chromium-driver
RUN apt update && apt install chromium-driver -y

# Use Poetry to install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Run app.py when the container launches
CMD ["python", "app.py"]
