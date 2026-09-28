import tarfile

import pytest

from pest import download_files
from pest.illustris_skirt_downloader import download_file, extract_tarball, get_simulation_name


class FakeResponse:
    def __init__(self, content: bytes, headers: dict, status_code: int = 200):
        self._content = content
        self.headers = headers
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_download_file_uses_content_disposition_filename(tmp_path, monkeypatch):
    def fake_get(url, headers, stream, timeout):
        assert headers == {"api-key": "secret"}
        return FakeResponse(b"tar-bytes", {"content-disposition": 'attachment; filename="skirt_images_sdss.99.tar"'})

    monkeypatch.setattr("pest.illustris_skirt_downloader.requests.get", fake_get)

    destination = download_file(
        "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar",
        tmp_path,
        api_key="secret",
    )

    assert destination == tmp_path / "TNG100-1" / "skirt_images_sdss.99.tar"
    assert destination.read_bytes() == b"tar-bytes"


def test_download_file_falls_back_to_url_basename(tmp_path, monkeypatch):
    def fake_get(url, headers, stream, timeout):
        return FakeResponse(b"tar-bytes", {})

    monkeypatch.setattr("pest.illustris_skirt_downloader.requests.get", fake_get)

    destination = download_file(
        "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar",
        tmp_path,
        api_key="secret",
    )

    assert destination.name == "skirt_images_sdss.99.tar"


def test_download_file_skips_existing_file(tmp_path, monkeypatch):
    existing = tmp_path / "TNG100-1" / "skirt_images_sdss.99.tar"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already-here")

    def fake_get(url, headers, stream, timeout):
        return FakeResponse(b"new-bytes", {})

    monkeypatch.setattr("pest.illustris_skirt_downloader.requests.get", fake_get)

    destination = download_file(
        "http://www.tng-project.org/api/TNG100-1/files/skirt_images_sdss.99.tar",
        tmp_path,
        api_key="secret",
    )

    assert destination.read_bytes() == b"already-here"


def test_download_files_downloads_each_url(tmp_path, monkeypatch):
    def fake_get(url, headers, stream, timeout):
        filename = url.rsplit("/", 1)[-1]
        return FakeResponse(filename.encode(), {})

    monkeypatch.setattr("pest.illustris_skirt_downloader.requests.get", fake_get)

    urls = [
        "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.95.tar",
        "http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.99.tar",
    ]
    destinations = download_files(urls, tmp_path, api_key="secret")

    assert [d.name for d in destinations] == ["skirt_images_sdss.95.tar", "skirt_images_sdss.99.tar"]
    for destination in destinations:
        assert destination.parent == tmp_path / "TNG50-1"
        assert destination.exists()


def test_get_simulation_name():
    assert get_simulation_name("http://www.tng-project.org/api/TNG50-1/files/skirt_images_sdss.95.tar") == "TNG50-1"
    assert (
        get_simulation_name("http://www.tng-project.org/api/Illustris-1/files/skirt_images_sdss.131.tar")
        == "Illustris-1"
    )


def test_get_simulation_name_invalid_url():
    with pytest.raises(ValueError):
        get_simulation_name("http://www.tng-project.org/not-an-api-url")


def test_extract_tarball(tmp_path):
    member_path = tmp_path / "member.txt"
    member_path.write_text("hello")

    tar_path = tmp_path / "archive.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(member_path, arcname="member.txt")
    member_path.unlink()

    destination = extract_tarball(tar_path)

    assert destination == tmp_path
    assert (tmp_path / "member.txt").read_text() == "hello"
    assert not tar_path.exists()


def test_get_illustris_api_key_from_env(monkeypatch):
    from pest import get_illustris_api_key

    monkeypatch.setenv("ILLUSTRIS_API_KEY", "env-secret")
    assert get_illustris_api_key() == "env-secret"


def test_get_illustris_api_key_missing(monkeypatch):
    from pest.illustris_skirt_downloader import get_illustris_api_key

    monkeypatch.delenv("ILLUSTRIS_API_KEY", raising=False)

    with pytest.raises(ValueError):
        get_illustris_api_key()
