import os
import io
import re
from typing import BinaryIO, Optional
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
import urllib3

from app.config import settings


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent invalid characters in MinIO object keys."""
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
    return clean_name or "document.bin"


class LocalFileStreamWrapper:
    """Wrapper around local file to match MinIO get_object stream API."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = open(file_path, "rb")

    def read(self, *args, **kwargs):
        return self._file.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._file.seek(*args, **kwargs)

    def close(self):
        self._file.close()

    def stream(self, chunk_size: int = 32768):
        self._file.seek(0)
        while True:
            chunk = self._file.read(chunk_size)
            if not chunk:
                break
            yield chunk


class MinIOStorageService:
    """Service for interacting with MinIO / S3 object storage with local filesystem fallback."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        self.endpoint = endpoint or settings.MINIO_ENDPOINT
        self.access_key = access_key or settings.MINIO_ROOT_USER
        self.secret_key = secret_key or settings.MINIO_ROOT_PASSWORD
        self.bucket_name = bucket_name or settings.MINIO_BUCKET_NAME
        self.secure = secure if secure is not None else settings.MINIO_SECURE
        self._client: Optional[Minio] = None
        self._is_available: Optional[bool] = None

    @property
    def client(self) -> Minio:
        """Lazily initialize the Minio client."""
        if self._client is None:
            self._client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                http_client=urllib3.PoolManager(
                    timeout=urllib3.Timeout(connect=1.0, read=2.0),
                    retries=urllib3.Retry(total=0),
                )
            )
        return self._client

    def ensure_bucket_exists(self) -> bool:
        """Check if the default storage bucket exists; if not, create it."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"[OK] Created MinIO bucket '{self.bucket_name}'")
            self._is_available = True
            return True
        except Exception:
            print(f"[INFO] MinIO server offline on {self.endpoint}. Active storage: local filesystem fallback.")
            self._is_available = False
            return False

    def upload_file(
        self,
        file_stream: BinaryIO,
        filename: str,
        file_hash: str,
        file_size: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        clean_name = sanitize_filename(filename)
        object_name = f"uploads/{file_hash[:16]}_{clean_name}"

        file_stream.seek(0)

        # If MinIO was determined offline, write directly to local disk without HTTP timeout
        if self._is_available is False:
            local_dir = os.path.join(os.getcwd(), "storage", "uploads")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{file_hash[:16]}_{clean_name}")
            with open(local_path, "wb") as f:
                f.write(file_stream.read())
            file_stream.seek(0)
            return object_name

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_stream,
                length=file_size,
                content_type=content_type,
            )
            file_stream.seek(0)
            self._is_available = True
            return object_name
        except Exception:
            self._is_available = False
            local_dir = os.path.join(os.getcwd(), "storage", "uploads")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, f"{file_hash[:16]}_{clean_name}")
            
            file_stream.seek(0)
            with open(local_path, "wb") as f:
                f.write(file_stream.read())
            file_stream.seek(0)
            return object_name

    def get_file_object(self, object_name: str):
        """Retrieve the raw object data stream from MinIO or local fallback."""
        if self._is_available:
            try:
                return self.client.get_object(self.bucket_name, object_name)
            except Exception:
                pass

        filename = os.path.basename(object_name)
        local_path = os.path.join(os.getcwd(), "storage", "uploads", filename)
        if os.path.exists(local_path):
            return LocalFileStreamWrapper(local_path)
        raise FileNotFoundError(f"Object '{object_name}' not found")

    def get_presigned_download_url(self, object_name: str, expires_seconds: int = 3600) -> str:
        """Generate a temporary, secure presigned download URL."""
        if self._is_available:
            try:
                return self.client.presigned_get_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    expires=timedelta(seconds=expires_seconds),
                )
            except Exception:
                pass
        return f"http://localhost:8000/api/v1/documents/download_fallback/{object_name}"

    def delete_file(self, object_name: str) -> None:
        """Delete an object from MinIO or local fallback."""
        if self._is_available:
            try:
                self.client.remove_object(self.bucket_name, object_name)
            except Exception:
                pass
        filename = os.path.basename(object_name)
        local_path = os.path.join(os.getcwd(), "storage", "uploads", filename)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

    def check_health(self) -> bool:
        """Verify that MinIO is reachable and responding."""
        if self._is_available is False:
            return False
        try:
            res = self.client.bucket_exists(self.bucket_name)
            self._is_available = res
            return res
        except Exception:
            self._is_available = False
            return False


minio_storage = MinIOStorageService()
