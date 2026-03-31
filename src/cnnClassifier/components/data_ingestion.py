import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Download the data snippet.
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading data from {self.config.bucket_name} to {self.config.local_data_file}")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            # Note: actual download URL would replace the bucket_name logic here if it was a URL.
            # E.g. request.urlretrieve(url=self.config.source_URL, filename=self.config.local_data_file)
            # For kaggle or aws, you might use their respectively Python APIs.
            logger.info("Add download logic here based on your dataset source (e.g. gdown, Kaggle API, or requests)")
            pass
        else:
            logger.info(f"File already exists with size: {get_size(Path(self.config.local_data_file))}")  

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        if os.path.exists(self.config.local_data_file):
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted files from {self.config.local_data_file} to {unzip_path}")
        else:
            logger.warning(f"Zip file not found at {self.config.local_data_file}")
