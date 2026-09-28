"""Downloader for Illustris/TNG SKIRT synthetic image tarballs."""

import argparse
import os
import re
import tarfile
from pathlib import Path

import requests

SKIRT_URLS = [
    "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.95.tar",
    "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.99.tar",
    "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.95.tar",
    "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar",
    "http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.131.tar",
    "http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.135.tar",
]


def get_illustris_api_key() -> str:
    """Return the API key for the Illustris/TNG API from the ILLUSTRIS_API_KEY environment variable."""
    if "ILLUSTRIS_API_KEY" in os.environ:
        return os.environ["ILLUSTRIS_API_KEY"]

    raise ValueError("No API key found. Please set the ILLUSTRIS_API_KEY environment variable.")


def get_simulation_name(url: str) -> str:
    """Extract the simulation name (e.g. 'TNG50-1') from a TNG/Illustris API URL."""
    match = re.search(r"/api/([^/]+)/", url)
    if not match:
        raise ValueError(f"Could not determine simulation name from URL: {url}")
    return match.group(1)


def download_file(url: str, output_path: Path, api_key: str, chunk_size: int = 1 << 20) -> Path:
    """Download a single file, honoring the server's suggested filename.

    The file is placed in a subdirectory of output_path named after its
    simulation (e.g. 'TNG50-1'), which is created if needed.

    Equivalent to `wget -nc --content-disposition`: the download is skipped if
    a file with the resolved filename already exists in the simulation directory.
    """
    output_path = Path(output_path) / get_simulation_name(url)
    output_path.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers={"api-key": api_key}, stream=True, timeout=60) as response:
        response.raise_for_status()

        filename = None
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            match = re.search(r'filename="?([^";]+)"?', content_disposition)
            if match:
                filename = match.group(1)
        if filename is None:
            filename = url.rsplit("/", 1)[-1]

        destination = output_path / filename
        if destination.exists():
            print(f"Skipping {filename} (already exists)")
            return destination

        print(f"Downloading {filename} ...")
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)

    return destination


def extract_tarball(tar_path: Path) -> Path:
    """Extract a tarball into its containing directory, delete it, and return that directory."""
    destination = tar_path.parent
    print(f"Extracting {tar_path.name} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(destination, filter="data")
    tar_path.unlink()
    return destination


def download_files(urls: list[str], output_path: str | Path, api_key: str | None = None) -> list[Path]:
    """Download a list of URLs into per-simulation subdirectories of output_path.

    Each file is placed in output_path/<simulation> (e.g. output_path/TNG50-1),
    skipping files that already exist.
    """
    api_key = api_key or get_illustris_api_key()
    output_path = Path(output_path)
    return [download_file(url, output_path, api_key) for url in urls]


def main() -> None:
    """CLI entry point: download the SKIRT tarballs to a given path."""
    parser = argparse.ArgumentParser(
        prog="pest-download-skirt",
        description="Download Illustris/TNG SKIRT synthetic image tarballs.",
    )
    parser.add_argument("path", help="Directory to download the files into.")
    parser.add_argument(
        "--urls-file",
        help="Optional text file with one URL per line. Defaults to the built-in SKIRT tarball list.",
    )
    args = parser.parse_args()

    if args.urls_file:
        with open(args.urls_file) as fh:
            urls = [line.strip() for line in fh if line.strip()]
    else:
        urls = SKIRT_URLS

    destinations = download_files(urls, args.path)
    for destination in destinations:
        extract_tarball(destination)


if __name__ == "__main__":
    main()
