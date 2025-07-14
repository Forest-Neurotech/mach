"""General utilities for downloading and caching files with integrity verification.

This module provides:
- Efficient file hashing compatible with Python 3.9+ (backport of hashlib.file_digest)
- Robust file downloading with progress bars and integrity verification
- Smart caching with automatic re-download on corruption
- Support for multiple hash algorithms (SHA1, SHA256, MD5, etc.)

Key Functions:
- file_digest(): Python 3.11+ compatible file hashing for older Python versions
- cached_download(): Download and cache files with integrity verification
- verify_file_integrity(): Verify file integrity using various hash algorithms

Optional dependencies:
- tqdm: For progress bars during downloads
"""

import hashlib
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import requests

if TYPE_CHECKING:
    from tqdm import tqdm

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

CACHE_DIR = Path.home() / ".cache" / "mach"


def file_digest(fileobj, digest: Union[str, Callable[[], "hashlib._Hash"]]) -> "hashlib._Hash":
    """Return a digest object that has been updated with contents of file object.

    This is a backport-compatible implementation of hashlib.file_digest()
    that works with Python 3.9+ and follows the same API as Python 3.11+.

    Args:
        fileobj: File-like object opened for reading in binary mode
        digest: Hash algorithm name as str, hash constructor, or callable that
            returns hash object

    Returns:
        Hash object with file contents

    Example:
        with open("file.bin", "rb") as f:
            hash_obj = file_digest(f, "sha256")
            print(hash_obj.hexdigest())
    """
    # Try to use native hashlib.file_digest if available (Python 3.11+)
    if sys.version_info >= (3, 11) and hasattr(hashlib, "file_digest"):
        return hashlib.file_digest(fileobj, digest)

    # Fallback implementation for Python 3.9 and 3.10
    # Get hash object from digest parameter
    if isinstance(digest, str):
        hash_obj = hashlib.new(digest)
    elif callable(digest):
        hash_obj = digest()
    else:
        msg = f"digest must be a string or callable, not {type(digest)}"
        raise TypeError(msg)

    # Efficient file reading with optimal buffer size and minimal memory allocation
    # Use 64KB buffer - good balance between memory usage and I/O efficiency
    buffer_size = 64 * 1024
    buf = bytearray(buffer_size)
    view = memoryview(buf)

    # Read file in chunks and update hash
    while True:
        bytes_read = fileobj.readinto(buf)
        if not bytes_read:
            break
        hash_obj.update(view[:bytes_read])

    return hash_obj


def verify_file_integrity(
    file_path: Path,
    expected_hash: str,
    digest: Union[str, Callable[[], "hashlib._Hash"]],
) -> bool:
    """Verify file integrity using specified hash algorithm.

    Args:
        file_path: Path to the file to verify
        expected_hash: Expected hash value
        digest: Hash algorithm to use, e.g. "sha256" or hashlib.sha256

    Returns:
        True if hash matches, False otherwise
    """
    if not file_path.exists():
        return False

    with open(file_path, "rb", buffering=0) as f:
        actual_hash = file_digest(f, digest).hexdigest()

    return actual_hash.lower() == expected_hash.lower()


def _verify_file(
    output_path: Path,
    expected_size: Optional[int],
    expected_hash: Optional[str],
    digest: Optional[Union[str, Callable[[], "hashlib._Hash"]]],
) -> bool:
    """Verify if existing file meets size and hash requirements.

    Returns:
        bool: True if file is valid, False if it should be re-downloaded
    """
    if not output_path.exists():
        return False

    if expected_size is not None and output_path.stat().st_size != expected_size:
        warnings.warn(f"File {output_path} has incorrect size", stacklevel=2)
        output_path.unlink()
        return False

    if (
        (expected_hash is not None)
        and (digest is not None)
        and not verify_file_integrity(output_path, expected_hash, digest)
    ):
        warnings.warn(f"File {output_path} failed integrity check", stacklevel=2)
        output_path.unlink()
        return False

    return True


def _download_with_progress(
    response: requests.Response,
    output_path: Path,
    chunk_size: int,
    total_size: int,
    show_progress: bool,
) -> None:
    """Download file content with optional progress bar."""
    if show_progress:
        assert tqdm is not None
        with (
            open(output_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {output_path.name}",
                disable=total_size == 0,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_file(
    url: str,
    output_path: Union[str, Path],
    timeout: int = 30,
    chunk_size: int = 1024 * 1024,  # 1MB
    *,
    overwrite: bool = False,
    expected_hash: Optional[str] = None,
    digest: Union[None, str, Callable[[], "hashlib._Hash"]] = None,
    expected_size: Optional[int] = None,
    show_progress: bool = (tqdm is not None),
) -> Path:
    """Download a file from a URL with optional progress bar and integrity verification.

    Args:
        url: URL to download
        output_path: Path where the file will be saved
        timeout: Connection timeout in seconds
        chunk_size: Size of chunks to download
        overwrite: Whether to overwrite existing files
        expected_hash: Expected hash value for integrity verification
        digest: Hash algorithm to use (default: "sha1")
        expected_size: Expected file size in bytes
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If download fails or integrity check fails
        ImportError: If show_progress=True but tqdm is not installed
    """
    if show_progress and (tqdm is None):
        raise ImportError("tqdm is required for progress bars. Install with: pip install tqdm")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing file
    if not overwrite and _verify_file(output_path, expected_size, expected_hash, digest):
        return output_path

    # Download the file
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if expected_size is not None and total_size != expected_size:
            warnings.warn(f"Server reports size {total_size}, expected {expected_size}", stacklevel=2)

        _download_with_progress(response, output_path, chunk_size, total_size, show_progress)

    except (OSError, requests.RequestException) as err:
        output_path.unlink(missing_ok=True)
        msg = f"Failed to download {url}: {err!s}"
        raise RuntimeError(msg) from err

    if not _verify_file(output_path, expected_size, expected_hash, digest):
        raise RuntimeError("Downloaded file failed either size or hash check")

    return output_path


def cached_download(
    url: str,
    cache_dir: Union[str, Path] = CACHE_DIR,
    filename: Optional[Union[str, Path]] = None,
    timeout: int = 30,
    *,
    overwrite: bool = False,
    expected_size: Optional[int] = None,
    expected_hash: Optional[str] = None,
    digest: Union[None, str, Callable[[], "hashlib._Hash"]] = None,
    show_progress: bool = (tqdm is not None),
) -> Path:
    """Download a file and cache it with optional integrity verification.

    Args:
        url: URL to download
        cache_dir: Directory to cache the file in
        filename: Name to save the file as (default: derived from URL)
            if absolute path is provided, it will be used as-is without cache-dir
        timeout: Connection timeout in seconds
        overwrite: Whether to overwrite existing files
        expected_hash: Expected hash value for integrity verification
        digest: Hash algorithm to use (default: "sha1")
        expected_size: Expected file size in bytes
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        Path to the cached file
    """
    # Get filename from URL if not provided
    if filename is None:
        cached_file = Path(cache_dir) / Path(url).name
    elif Path(filename).is_absolute():
        cached_file = Path(filename)
    else:
        cached_file = Path(cache_dir) / filename

    assert isinstance(cached_file, Path), "cached_file must be a Path"
    assert cached_file.is_absolute(), "cached_file must be an absolute path"

    # Download with optional integrity verification
    return download_file(
        url=url,
        output_path=cached_file,
        timeout=timeout,
        overwrite=overwrite,
        expected_size=expected_size,
        expected_hash=expected_hash,
        digest=digest,
        show_progress=show_progress,
    )
