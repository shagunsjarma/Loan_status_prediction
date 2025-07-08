from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

if __name__ == "__main__":
    logging.info("Starting the training pipeline...")
    # Data Ingestion
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Model Training
    trainer = ModelTrainer()
    accuracy = trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model training completed. Test Accuracy: {accuracy}")
