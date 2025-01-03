import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor,
                              AdaBoostRegressor
                              )
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.logger import logger
from src.exception import CustomException
from src.utils import save_obj, eval_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array): 
        '''
        This function trains the model on the train data and evaluates it on the test data.
        The best model is saved as a pkl file.'''
        try:
            logger.info('Splitting the data into train and test sets')
            # Split train_array into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                train_array[:, :-1], 
                train_array[:, -1], 
                test_size=0.2, 
                random_state=42
            )

            # Use test_array directly as the test set
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models={
                'RandomForestRegressor':RandomForestRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'LinearRegression':LinearRegression(),
                'SVR':SVR(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=0)
            }

            #The hyperparameters for the models are defined below.
            
            params={
                'RandomForestRegressor':{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                'GradientBoostingRegressor':{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                'AdaBoostRegressor':{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                'LinearRegression':{

                },

                "SVR":{
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']
                },

                'KNeighborsRegressor':{
                    'n_neighbors': [2,3,4,5,6,7,8,9,10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },

                'DecisionTreeRegressor': {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
            
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            # Evaluating the models
            model_report=eval_models(
                model=models,
                param=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )

            # Finding the best model
            best_model_score=max(model_report.values())
            best_model_name=[key for key in model_report if model_report[key]==best_model_score][0]

            best_model=models[best_model_name]
            if best_model_score<0.60:
                raise CustomException('No best model found',sys)
            logger.info(f'The best model is: "{best_model_name}" with a score of {best_model_score}')

            # Saving the best model as pkl file
            save_obj(
                obj=best_model, 
                file_path=self.model_trainer_config.trained_model_file_path
            )
            logger.info(f'The best model has been saved as {self.model_trainer_config.trained_model_file_path}')
            
            # Evaluating the best model
            predicted_target=best_model.predict(X_test)
            r2_score_value=r2_score(y_test,predicted_target)

            return r2_score_value
            
        except Exception as e:
            raise CustomException(e,sys)



