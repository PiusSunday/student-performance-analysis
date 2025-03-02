# Import the necessary libraries and modules
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..exception import CustomException
from ..logger import logging
from ..utils import save_object


# Define a data class to store the path for saving the preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts", "preprocessor.pkl"
    )  # Path to save the preprocessor object


# Define the DataTransformation class to handle data transformation
class DataTransformation:
    def __init__(self):
        # Initialize the data transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation pipeline.

        Returns:
            preprocessor: A ColumnTransformer object for preprocessing numerical and categorical features.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create a pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Impute missing values with the median
                    ("scaler", StandardScaler()),  # Scale the numerical features
                ]
            )

            # Create a pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Impute missing values with the most frequent value
                    (
                        "one_hot_encoder",
                        OneHotEncoder(),
                    ),  # Encode categorical features using one-hot encoding
                    (
                        "scaler",
                        StandardScaler(with_mean=False),
                    ),  # Scale the categorical features
                ]
            )

            # Log the categorical and numerical columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns,
                    ),  # Apply numerical pipeline to numerical columns
                    (
                        "cat_pipelines",
                        cat_pipeline,
                        categorical_columns,
                    ),  # Apply categorical pipeline to categorical columns
                ]
            )

            return preprocessor

        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for initiating the data transformation process.

        Args:
            train_path (str): Path to the training data file.
            Test_path (str): Path to the testing data file.

        Returns:
            train_arr (np.array): Transformed training data array.
            Test_arr (np.array): Transformed testing data array.
            Preprocessor_obj_file_path (str): Path to the saved preprocessor object.
        """
        try:
            # Read the training and testing data into pandas DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target feature for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessing object on the training and testing data
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed input features with the target feature
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object to a file
            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Return the transformed data arrays and the path to the saved preprocessor object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
