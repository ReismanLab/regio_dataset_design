import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_zip(url, extract_to="extracted"):

    # Download the zip file
    zip_filename = "model_validation.zip"

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(zip_filename, "wb") as zip_file, tqdm(
        desc=f"model_validation Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            zip_file.write(chunk)
            progress_bar.update(len(chunk))
    
    # Step 2: Unzip the file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def merge_folders(src_folder, dest_folder):

    for root, dirs, files in os.walk(src_folder):

        rel_path = os.path.relpath(root, src_folder)
        dest_path = os.path.join(dest_folder, rel_path)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            if not os.path.exists(dest_file):  # Only copy if file does not exist
                shutil.copy2(src_file, dest_file)
            else:
                print(f"Skipped (already exists): {dest_file}")

# Set the source and destination folders
src_folder = 'extracted/model_validation'
dest_folder = '../../results/model_validation'

download_zip("https://zenodo.org/records/14003333/files/model_validation.zip?download=1")
merge_folders(src_folder, dest_folder)

os.remove('model_validation.zip')
shutil.rmtree('extracted')

