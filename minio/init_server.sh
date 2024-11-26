#!/bin/bash

# Start the MinIO server in the background
/minio server /data &

# Wait for the server to start
sleep 5

# Run the initialization script to create buckets
python3 /app/init_minio.py

# Keep the container running by foregrounding the MinIO server
fg