import io
import hashlib
from app.core.hashing import calculate_stream_sha256


def test_calculate_stream_sha256():
    """Verify SHA-256 calculation matches hashlib.sha256 standard behavior."""
    content = b"Antigravity Document Processing Test Content 12345"
    stream = io.BytesIO(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    expected_size = len(content)

    calculated_hash, size = calculate_stream_sha256(stream)

    assert calculated_hash == expected_hash
    assert size == expected_size
    # Ensure stream position was reset to 0
    assert stream.tell() == 0
    assert stream.read() == content
