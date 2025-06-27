from pathlib import Path

from ultralytics.utils.downloads import download

# Download labels
segments = True  # segment or box labels
dir = Path(".")  # dataset root dir
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
# labels
urls = [
    url + ("coco2017labels-segments.zip" if segments else "coco2017labels.zip")]
download(urls, dir=dir.parent)
# Download data
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
    "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
    # 7G, 41k images (optional)
    "http://images.cocodataset.org/zips/test2017.zip",
]
download(urls, dir=dir / "images", threads=3)
