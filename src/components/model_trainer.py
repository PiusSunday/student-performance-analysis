# Import the necessary libraries and modules
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ..exception import CustomException
from ..logger import logging
from ..utils import evaluate_models, save_object


# Define a data class to store the path for saving the trained model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts", "model.pkl"
    )  # Path to save the trained model


# Define the ModelTrainer class to handle model training
class ModelTrainer:
    def __init__(self):
        # Initialize the model trainer configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        This function is responsible for initiating the model training process.

        Args:
            train_array (np.array): Training data array containing features and target.
            test_array (np.array): Testing data array containing features and target.

        Returns:
            r2_square (float): R-squared score of the best model on the test data.
        """
        try:
            # Log the start of splitting training and test data
            logging.info("Split training and test input data")

            # Split the input arrays into features (X) and target (y) for training and testing
            X_train, y_train, X_test, y_test = (
                train_array[
                    :, :-1
                ],  # Features for training (all columns except the last)
                train_array[:, -1],  # Target for training (last column)
                test_array[
                    :, :-1
                ],  # Features for testing (all columns except the last)
                test_array[:, -1],  # Target for testing (last column)
            )

            # Define a dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),  # Random Forest Regressor
                "Decision Tree": DecisionTreeRegressor(),  # Decision Tree Regressor
                "Gradient Boosting": GradientBoostingRegressor(),  # Gradient Boosting Regressor
                "Linear Regression": LinearRegression(),  # Linear Regression
                "XGBRegressor": XGBRegressor(),  # XGBoost Regressor
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False
                ),  # CatBoost Regressor
                "AdaBoost Regressor": AdaBoostRegressor(),  # AdaBoost Regressor
            }

            # Define hyperparameters for each model to tune during evaluation
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],  # Criteria for splitting
                },
                "Random Forest": {
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                    ]  # Number of trees in the forest
                },
                "Gradient Boosting": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001,
                    ],  # Learning rate for boosting
                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                    ],  # Fraction of samples used for fitting
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                    ],  # Number of boosting stages
                },
                "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
                "XGBRegressor": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001,
                    ],  # Learning rate for XGBoost
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                    ],  # Number of boosting rounds
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],  # Depth of the tree
                    "learning_rate": [0.01, 0.05, 0.1],  # Learning rate for CatBoost
                    "iterations": [30, 50, 100],  # Number of iterations
                },
                "AdaBoost Regressor": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.5,
                        0.001,
                    ],  # Learning rate for AdaBoost
                    "n_estimators": [8, 16, 32, 64, 128, 256],  # Number of estimators
                },
            }

            # Evaluate models using the `evaluate_models` utility function
            model_report: dict = evaluate_models(
                X_train=X_train,  # Training features
                y_train=y_train,  # Training target
                X_test=X_test,  # Testing features
                y_test=y_test,  # Testing target
                models=models,  # Dictionary of models
                param=params,  # Dictionary of hyperparameters
            )

            # Get the best model score from the evaluation results
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the evaluation results
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]  # Retrieve the best model

            # Check if the best model score is below the threshold (0.6)
            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found", sys
                )  # Raise exception if no model meets the threshold
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,  # Path to save the model
                obj=best_model,  # Best model object
            )

            # Make predictions using the best model on the test data
            predicted = best_model.predict(X_test)

            # Calculate the R-squared score for the predictions
            r2_square = r2_score(y_test, predicted)
            return r2_square  # Return the R-squared score

        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
