from minio import Minio
import time

def init_minio():
    # Initialize MinIO client
    client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

    # List of buckets to create
    buckets = ["models", "data", "results"]
    
    for bucket in buckets:
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                print(f"Created bucket: {bucket}")
            else:
                print(f"Bucket already exists: {bucket}")
        except Exception as e:
            print(f"Error creating bucket {bucket}: {e}")

    return client

if __name__ == "__main__":
    # Small delay to ensure MinIO is fully ready
    time.sleep(2)
    init_minio()