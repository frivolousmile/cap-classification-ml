import io, zipfile, urllib.request
from pathlib import Path

URL = "https://github.com/ravee360/Cap-detection/archive/refs/heads/main.zip"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"

def main():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "cap_detection.zip"
    print("Downloading public cap-detection dataset...")
    urllib.request.urlretrieve(URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    extracted = next(DATA_DIR.glob("Cap-detection-*"))
    target = DATA_DIR / "source"
    if target.exists():
        import shutil
        shutil.rmtree(target)
    extracted.rename(target)
    zip_path.unlink(missing_ok=True)
    print("Dataset ready at:", target)
    print("Expected structure: dataset/source/split/images and dataset/source/split/annotations")

if __name__ == "__main__":
    main()
