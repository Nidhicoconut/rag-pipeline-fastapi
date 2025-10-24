# File: Dockerfile

# 1. Start from an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (for 'pypdf' and other libraries)
# We update apt-get and install 'poppler-utils' which is needed by pypdf
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file and install Python libraries
# This is done in a separate step to leverage Docker's layer caching
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire application code into the container
COPY . .

# 6. Expose the port your app runs on
EXPOSE 8000

# 7. Define the command to run your application
# This runs uvicorn directly (not with --reload) for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]