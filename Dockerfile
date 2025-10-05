# Use a minimal Python image
FROM python:3.13.7

# Set environment variables for better output (optional)
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    libcairo2-dev \
    # Clean up to keep the image size small
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for efficient caching)
COPY requirements.txt .

# Install dependencies (using --no-cache-dir to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Inform Docker that the container will listen on this port
EXPOSE 8080

# Command to run the application when the container starts
CMD ["python", "main.py"]