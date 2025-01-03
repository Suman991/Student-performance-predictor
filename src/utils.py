import os
import sys
import dill
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Add the parent directory of the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.logger import logger
from src.exception import CustomException


def save_obj(obj, file_path):
    '''
    This function is responsible for saving the object as a pickle file
    '''
    try:
       dir_path = os.path.dirname(file_path)
       os.makedirs(dir_path, exist_ok=True)
       with open(file_path, 'wb') as f:
           dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e,sys)
    

def eval_models(model,param,X_train,y_train,X_val,y_val):
    '''
    This function is responsible for evaluating the models
    '''
    try:
        report={}
        for model_key,param_key in zip(model.keys(),param.keys()):

            gs=GridSearchCV(model[model_key],param[param_key],cv=3)
            gs.fit(X_train,y_train)

            model[model_key].set_params(**gs.best_params_)
            model[model_key].fit(X_train,y_train)    #Training the model
          
            #Predicting the target variable for the training and validation datasets
            y_train_pred=model[model_key].predict(X_train)
            y_val_pred=model[model_key].predict(X_val)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_val,y_val_pred)

            report[model_key]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    '''
    This function is responsible for loading the object from the pickle file
    '''
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)