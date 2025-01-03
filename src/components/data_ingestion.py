import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Add the parent directory of the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.logger import logger
from src.exception import CustomException

#No need to add parent directory of src directory to the Python path
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #contains the paths for the raw, train and test data
    
    def initiate_data_ingestion(self):
        logger.info("Initiating data ingestion")
        try:
            # Read the dataset
            df=pd.read_csv("notebook\data\StudentsPerformance.csv")
            logger.info("Dataset is read successfully as dataframe")

            # Create the directory("artifacts") if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data taken from the source
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logger.info("Train test split is initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logger.info("Data ingestion is completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        



if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation_obj=DataTransformation()
    train_arr,test_arr,_=data_transformation_obj.initiate_data_transformation(train_data,test_data)

    model_trainer_obj=ModelTrainer()
    best_score=model_trainer_obj.initiate_model_trainer(train_arr,test_arr)
    print(f"Best model score is {best_score}")
