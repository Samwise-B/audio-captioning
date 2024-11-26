import os
from minio import Minio
from minio.error import S3Error


def create_buckets():
    # MinIO configuration from environment variables
    minio_address = os.getenv("MINIO_ADDRESS", "localhost:9000")
    minio_access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
    minio_secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

    # List of buckets to initialize
    buckets = ["bucket1", "bucket2", "bucket3"]

    # Create MinIO client
    client = Minio(
        minio_address,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    # Create buckets if they don't already exist
    for bucket in buckets:
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                print(f"Bucket '{bucket}' created successfully.")
            else:
                print(f"Bucket '{bucket}' already exists.")
        except S3Error as e:
            print(f"Error creating bucket '{bucket}': {e}")


if __name__ == "__main__":
    create_buckets()
