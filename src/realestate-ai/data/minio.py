import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "realestate")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")

# --- S3 / MinIO helpers ---


def get_s3_client():
    """Return a configured boto3 S3 client pointing to MinIO."""
    session = boto3.Session()
    client = session.client(
        service_name="s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    return client


def ensure_bucket(bucket_name: str = MINIO_BUCKET):
    """Create the bucket if it doesn't exist (idempotent)."""
    s3 = get_s3_client()
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        # Bucket does not exist or is inaccessible, attempt to create
        try:
            s3.create_bucket(Bucket=bucket_name)
        except ClientError:
            # If bucket already exists but is owned by someone else, this will fail.
            raise


def upload_html(bucket_name: str, key: str, html_bytes: bytes):
    """Upload raw HTML bytes to MinIO (S3-compatible)."""
    s3 = get_s3_client()
    ensure_bucket(bucket_name)
    s3.put_object(Bucket=bucket_name, Key=key, Body=html_bytes, ContentType="text/html")


def load_html(bucket_name: str, key: str) -> bytes:
    """Download raw HTML bytes from MinIO (S3-compatible)."""
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket_name, Key=key)
    return response["Body"].read()


def get_bucket_name():
    return MINIO_BUCKET
