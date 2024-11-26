# Use the official MinIO image as the base
FROM minio/minio:latest

# Install Python for running the init.py script
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Copy initialization scripts
COPY init.py /app/init.py
COPY init.sh /app/init.sh

# Make the init.sh script executable
RUN chmod +x /app/init.sh

# Expose MinIO default ports
EXPOSE 9000 9001

# Entry point to initialize buckets and start the MinIO server
ENTRYPOINT ["/app/init_server.sh"]
