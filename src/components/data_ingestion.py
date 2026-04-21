import os
import sys
import pandas as pd
from dataclasses import dataclass

import sklearn

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts",'training_data.csv')
    test_data_path: str = os.path.join("artifacts",'test_data.csv')
    raw_data_path: str = os.path.join("artifacts",'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def data_ingestion_start(self,data_file_name):
        try:
            df = pd.read_csv(data_file_name)
            logging.info('Reading the data is completed')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f'The raw data file is saved at {self.data_ingestion_config.raw_data_path}')
            train_data,test_data= sklearn.model_selection.train_test_split(df,test_size=0.2,random_state=42)
            logging.info(f'The splitting is completed')
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info(f'The splitting is completed and the data have been saved in the artifacts folder !!!')
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            CustomException(e,sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.data_ingestion_start('dataset/riceClassification.csv')
    print('The data ingestion script is fine')
