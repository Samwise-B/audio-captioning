from minio import Minio
from io import BytesIO
import time

class MinioClientWrapper:
    def __init__(self):
        self.client = Minio(
            "minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        self._create_default_buckets()


    def _create_default_buckets(self):
        buckets = ["models", "data", "results"]
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    print(f"Created bucket: {bucket}")
                else:
                    print(f"Bucket already exists: {bucket}")
            except Exception as e:
                print(f"Error creating bucket {bucket}: {e}")

    def save_weights(self, weights_file, model_name, version):
        """
        Simple function to save weights to MinIO.
        
        Example path: models/my_model_v1.pth
        """
        try:
            path = f"{model_name}_{version}.pth"
            self.client.fput_object("models", path, weights_file)
            print(f"Saved weights: {path}")
        except Exception as e:
            print(f"Error saving weights: {e}")

    def load_weights(self, model_name, version):
        """
        Load weights directly into memory
        Returns: BytesIO object containing the weights
        """
        try:
            path = f"{model_name}_{version}.pth"
            data = self.client.get_object("models", path)
            weights = BytesIO(data.read())
            print(f"Successfully loaded weights for {model_name} v{version} into memory")
            return weights
        except Exception as e:
            print(f"Error loading weights: {e}")
            return None

if __name__ == "__main__":
    time.sleep(2)
    minio = MinioClientWrapper