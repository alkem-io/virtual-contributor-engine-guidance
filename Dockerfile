# Use an official Python runtime as a parent image
FROM python:3-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# install chromium-driver
RUN apt update && apt install chromium-driver -y

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "app.py"]
