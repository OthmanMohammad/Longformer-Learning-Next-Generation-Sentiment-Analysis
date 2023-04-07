from data.data_loader import download_and_extract_dataset

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dest_dir = "data"
download_and_extract_dataset(url, dest_dir)
