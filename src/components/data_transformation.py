import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Add the parent directory of the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.logger import logger
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''
    This class is responsible for creating the data transformation object
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            numerical_columns=['reading score','writing score']
            categorical_columns=['gender', 'race/ethnicity',
                                  'parental level of education', 
                                  'lunch','test preparation course'
                                ]

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False)) # Set with_mean=False for sparse matrices
                ]
            )
            logger.info('Numerical columns standardization pipeline created')

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder())
                ]
            )
            logger.info('Categorical columns encoding pipeline created')

            preprocssor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocssor
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function is responsible for reading the train and test data 
        and applying the preprocessing object on it
        '''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logger.info('Train and test data read successfully')

            logger.info('Obtaining preprocessing object')
            preprocessing_obj=self.get_data_transformation_object()

            # Splitting the data into "input" and "target" features both for training and testing data
            target_column_name='math score' 

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            logger.info('Applying Preprocessing object on training & testing dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Combining the input and target features
            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logger.info('Preprocessing object applied successfully on training & testing dataframe')


            save_obj(
                obj=preprocessing_obj,
                file_path=self.data_transformation_config.preprocessor_obj_file_path
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            ) 

        except Exception as e:
            raise CustomException(e,sys)
        


