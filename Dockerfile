# Use a minimal Python image
FROM python:3.13.7

# Set environment variables for better output (optional)
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /usr/src/app

# Copy dependency file first (for efficient caching)
COPY requirements.txt .

# Install dependencies (using --no-cache-dir to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Inform Docker that the container will listen on this port
EXPOSE 8000

# Command to run the application when the container starts
CMD ["python", "main.py"]