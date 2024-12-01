import os, wget, tarfile
from tqdm import tqdm


def unpack_archive(archive_path, extract_to):
    with tarfile.open(archive_path, "r") as tar:
        for member in tqdm(tar.getmembers()):
            extracted_path = f"{extract_to}/{member.name}"
            if not os.path.exists(extracted_path):
                tar.extract(member, extract_to)


def download_and_extract_dataset(dataset_url, dataset_dir, dataset_filename=None):
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_filename = dataset_filename or os.path.basename(dataset_url)
    archive_path = f"{dataset_dir}/{dataset_filename}"
    if not os.path.exists(archive_path):
        wget.download(dataset_url, archive_path)
    unpack_archive(archive_path, dataset_dir)


dataset_dir = "dataset"
dataset_url = "https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar"
download_and_extract_dataset(dataset_url, dataset_dir)
