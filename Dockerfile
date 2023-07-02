# Use an official Python runtime as a parent image
FROM python:3-slim-bookworm

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# install chromium-driver
RUN apt update && apt install chromium-driver -y
# uncomment the following to install Google Chrome
#RUN curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add
#RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
#RUN apt-get -y update
#RUN apt-get -y install google-chrome-stable

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]