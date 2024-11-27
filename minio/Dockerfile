# Use a Python base image
FROM python:3.10-slim

# Install MinIO server
RUN apt-get update && apt-get install -y wget \
    && wget https://dl.min.io/server/minio/release/linux-amd64/minio \
    && chmod +x minio \
    && mv minio /usr/local/bin/

# Install required Python packages
RUN pip install minio

# Set working directory
WORKDIR /app

# Copy initialization scripts
COPY init_minio.py /app/init.py
COPY init_server.sh /app/init.sh

# Make the init.sh script executable
RUN chmod +x /app/init.sh

# Expose MinIO default ports
EXPOSE 9000 9001

# Entry point to initialize buckets and start the MinIO server
ENTRYPOINT ["/app/init.sh"]
