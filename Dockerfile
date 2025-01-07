# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies for compiling native extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libz-dev \
    liblzma-dev \
    libsnappy-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if needed, for example for Jupyter)
EXPOSE 8888

# Default command: run an interactive shell session
CMD ["bash"]
