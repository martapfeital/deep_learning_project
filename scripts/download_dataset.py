import os
import zipfile
import gdown

FILE_ID = "1No8kSNf_-JYXHtKGCH1anFi191HPZtrA"

DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "wikiart.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "wikiart")

def download_file(file_id, dest_path):
    print("Downloading dataset...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    print("Download finished.")

def extract_zip(zip_path, extract_to):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACT_DIR):
        print("Dataset already exists. Nothing to do.")
        return

    download_file(FILE_ID, ZIP_PATH)
    extract_zip(ZIP_PATH, DATA_DIR)

    os.remove(ZIP_PATH)
    print("Dataset ready in:", EXTRACT_DIR)

if __name__ == "__main__":
    main()