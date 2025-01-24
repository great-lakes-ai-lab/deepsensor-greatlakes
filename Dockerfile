# Use an official Python runtime as a parent image (Python 3.12)
FROM python:3.12-slim

# Install system dependencies for compiling native extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libz-dev \
    liblzma-dev \
    libsnappy-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    curl  # Ensure curl is available for fetching extra packages if needed

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install dependencies and the package in editable mode
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Expose port (if needed, for example for Jupyter)
EXPOSE 8888

# Create a non-root user for Jupyter
RUN useradd -m jupyter_user
USER jupyter_user

# Set up Jupyter configuration
RUN mkdir -p /home/jupyter_user/.jupyter && \
    jupyter notebook --generate-config

# Start Jupyter notebook instead of bash
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
