from __future__ import annotations

import concurrent.futures as cf
import sys
import zipfile
from pathlib import Path
from typing import List
from urllib.request import Request, urlopen

from tqdm import tqdm  # pip install tqdm

# ----------------------------- #
# Utility
# ----------------------------- #
_CHUNK = 1 << 14  # 16 KiB


def _download_one(url: str, out_path: Path) -> None:
    """Stream-download *url* to *out_path* with progress bar."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(out_path, "wb") as f, tqdm(
        total=int(r.headers.get("Content-Length", 0)),
        unit="B",
        unit_scale=True,
        desc=out_path.name,
        leave=False,
    ) as bar:
        while True:
            chunk = r.read(_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))


def _maybe_unzip(path: Path, remove_zip: bool = False) -> None:
    if path.suffix.lower() != ".zip":
        return
    with zipfile.ZipFile(path) as zf:
        zf.extractall(path.parent)
    if remove_zip:
        path.unlink()


def download_many(urls: List[str], dst_dir: Path, threads: int = 3) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    futures = []
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        for url in urls:
            fname = url.split("/")[-1]
            out = dst_dir / fname
            futures.append(ex.submit(_download_one, url, out))
        # 等全部下載完
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Downloading"):
            fut.result()
    # 解壓
    for zip_path in dst_dir.glob("*.zip"):
        _maybe_unzip(zip_path, remove_zip=False)


# ----------------------------- #
# Main
# ----------------------------- #
def main(segment_labels: bool = True) -> None:
    root = Path(".").resolve()
    base = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    label_zip = "coco2017labels-segments.zip" if segment_labels else "coco2017labels.zip"
    label_urls = [base + label_zip]

    coco_urls = [
        "http://images.cocodataset.org/zips/train2017.zip",  # 19 GB
        "http://images.cocodataset.org/zips/val2017.zip",    # 1 GB
        "http://images.cocodataset.org/zips/test2017.zip",   # 7 GB
    ]

    print("[1/2] Downloading label zips …")
    download_many(label_urls, root.parent)

    print("[2/2] Downloading image zips …")
    download_many(coco_urls, root / "images", threads=3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload aborted.", file=sys.stderr)
        sys.exit(1)
