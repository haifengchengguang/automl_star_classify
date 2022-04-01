import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

data_path= "automl_train_data.csv"
df=pd.read_csv(data_path) # path to colab notebook #replace this with the path to your dataset

bd_data = TabularDataset(data_path)
df_train,df_test=train_test_split(df,test_size=0.2,random_state=1)
#test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
save_path = './BDmodels/'
predictor = TabularPredictor(label='subclass',path=save_path).fit(train_data=df_train)
#predictions = predictor.predict(test_data)
