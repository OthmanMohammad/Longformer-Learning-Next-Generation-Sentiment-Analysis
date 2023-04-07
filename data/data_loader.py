import os
import tarfile
import urllib.request

def download_and_extract_dataset(url, dest_dir):
    filename = os.path.join(dest_dir, "aclImdb_v1.tar.gz")

    # Download the dataset
    urllib.request.urlretrieve(url, filename)

    # Extract the dataset
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(dest_dir)

    # Remove the downloaded archive
    os.remove(filename)
