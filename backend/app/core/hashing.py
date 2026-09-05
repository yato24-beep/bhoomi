import hashlib
from typing import BinaryIO, Tuple


def calculate_stream_sha256(file_stream: BinaryIO, chunk_size: int = 65536) -> Tuple[str, int]:
    """Calculate the SHA-256 hash and byte size of a file-like binary stream.
    
    Reads in 64KB chunks to maintain a minimal memory footprint for large files.
    Automatically resets the stream position back to 0 after reading.
    
    Args:
        file_stream: Binary file-like stream (e.g. UploadFile.file or open file).
        chunk_size: Number of bytes to read per iteration (default: 64 KB).
        
    Returns:
        Tuple of (sha256_hex_string, total_file_size_bytes)
    """
    hasher = hashlib.sha256()
    total_bytes = 0

    # Ensure reading starts from the beginning of the file
    file_stream.seek(0)

    while True:
        chunk = file_stream.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
        total_bytes += len(chunk)

    # Reset file pointer back to start so subsequent operations can read it
    file_stream.seek(0)

    return hasher.hexdigest(), total_bytes
