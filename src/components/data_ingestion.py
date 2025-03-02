# Import the necessary libraries and modules
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ..components.data_transformation import DataTransformation
from ..components.model_trainer import ModelTrainer
from ..exception import CustomException
from ..logger import logging


# Define a data class to store configuration paths for data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )  # Path to save the training data
    test_data_path: str = os.path.join(
        "artifacts", "test.csv"
    )  # Path to save the testing data
    raw_data_path: str = os.path.join(
        "artifacts", "data.csv"
    )  # Path to save the raw data


# Define the DataIngestion class to handle the data ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log the entry into the data ingestion method
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from the specified path into a pandas DataFrame
            df = pd.read_csv(
                "/Users/sunnythesage/PythonProjects/Data-Science-BootCamp/Datasets/student-performance-data.csv"
            )
            logging.info("Read the dataset as dataframe")

            # Create the directory for saving the data if it doesn't exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Log the initiation of the train-test split
            logging.info("Train test split initiated")
            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training data to the specified path
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            # Save the testing data to the specified path
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            # Log the completion of the data ingestion process
            logging.info("Ingestion of the data is completed")

            # Return the paths to the training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)


# Main block to execute the data ingestion, transformation, and model training
if __name__ == "__main__":
    # Initialize the DataIngestion object
    obj = DataIngestion()

    # Perform data ingestion and get the paths to the training and testing data
    train_data, test_data = obj.initiate_data_ingestion()

    # Initialize the DataTransformation object
    data_transformation = DataTransformation()

    # Perform data transformation and get the transformed data arrays
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    # Initialize the ModelTrainer object
    model_trainer = ModelTrainer()

    # Perform model training and print the results
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
