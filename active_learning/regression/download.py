import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_unzip_move_and_cleanup(url, name):

    # Step 1: Download the zip file with a progress bar
    zip_filename = f"{name}.zip"

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(zip_filename, "wb") as zip_file, tqdm(
        desc=f"{name} Downloading",
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
        zip_ref.extractall("temp_extracted")
    
    shutil.move(f"temp_extracted/{name}", f"../../results/active_learning/regression/{name}")
    shutil.rmtree("temp_extracted")
    
    # Step 4: Delete the zip file
    os.remove(zip_filename)

download_unzip_move_and_cleanup("https://zenodo.org/records/14003333/files/clean_run.zip?download=1", "clean_run")
download_unzip_move_and_cleanup("https://zenodo.org/records/14003333/files/clean_run.zip?download=1", "clean_run")
download_unzip_move_and_cleanup("https://zenodo.org/records/14003333/files/experimental.zip?download=1", "experimental")