# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)pip install -U scikit-learn
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
# Classifier Libraries
# Other Libraries
from sklearn.model_selection import train_test_split

from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
from collections import defaultdict

from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

if __name__=="__main__":
    def train(n_factors=25,n_epochs=25,lr_all=0.007,reg_all=0.2):
        with mlflow.start_run(run_name="SVD++"):

            df=pd.read_csv('transaction_data.csv')
    
            mlflow.log_artifact('transaction_data.csv')

            df=df.replace("-","",regex=True)
           
            max_tr=df.total_transaction.max()
            min_tr=df.total_transaction.min()

            reader=Reader(rating_scale=(min_tr,max_tr))
            data=Dataset.load_from_df(df[["customer_id","product_code","total_transaction"]],reader)

            trainset,testset=train_test_split(data,test_size=0.25,random_state=121)
           
            from surprise import SVDpp
            SVDpp =SVDpp(n_factors=n_factors,n_epochs=n_epochs,lr_all=lr_all,reg_all=reg_all)

            mlflow.log_param('n_factors',n_factors)
            mlflow.log_param('n_epocsh',n_epochs)
            mlflow.log_param('lr_all',lr_all)
            mlflow.log_param('reg_all',reg_all)

            #train
            SVDpp.fit(trainset)
            #test
            SVDpp_pred=SVDpp.test(testset)

            #save the model as a pickle in a file
            joblib.dump(SVDpp,'SVDpp_model.pkl')

            #load the model from the file
            SVDpp_from_joblib=joblib.load('SVDpp_model.pkl')

            #test
            SVDpp_pred=SVDpp_from_joblib.test(testset)

            RMSE=accuracy.rmse(SVDpp_pred,verbose=False)
            MAE=accuracy.mae(SVDpp_pred,verbose=False)
            

            print("SVD++ model (n_factors=%f,n_epochs=%f) : "% (SVDpp.n_factors, SVDpp.n_epochs))
            print(f'model RMSE is: {RMSE}')
            print(f'model MAE is: {MAE}')

            mlflow.log_metric("RMSE",RMSE)
            mlflow.log_metric("MAE",MAE)
            mlflow.sklearn.log_model(SVDpp,'Recsys')

train()

