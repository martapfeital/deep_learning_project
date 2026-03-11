import os
import zipfile
import requests

DATASET_URL = "https://liveeduisegiunl-my.sharepoint.com/:u:/g/personal/amarques_novaims_unl_pt/IQCcfwZTCUBRSKZg4wIQ9DOSAdA4fqruEHl5Ssx9dAGoprc?e=WSWtKf"

DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "wikiart.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "wikiart")

def download_file(url, dest_path):
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

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

    download_file(DATASET_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, DATA_DIR)

    os.remove(ZIP_PATH)
    print("Dataset ready in:", EXTRACT_DIR)

if __name__ == "__main__":
    main()