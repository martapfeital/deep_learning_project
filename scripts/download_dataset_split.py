import os
import zipfile
import gdown

print("Starting download script...")

FILE_ID = "19aGvVRnukcueArLumx6GFQCmkX4ipXj2"
ZIP_PATH = "data_split.zip"
DEST_ROOT = "data_split"

os.makedirs(DEST_ROOT, exist_ok=True)

if not os.listdir(DEST_ROOT):  
    print(f"{DEST_ROOT} is empty. Downloading...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)
    print("Download finished. Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DEST_ROOT)

    os.remove(ZIP_PATH)
    print(f"Dataset ready in '{DEST_ROOT}'")
else:
    print(f"Dataset already exists in '{DEST_ROOT}'")