import numpy as npy
import pandas as pds
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

#load data from .csv
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pds.read_csv(dataset_url, sep=';')

#separate target feature
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

#Standarize the data (already handled by CVPipeline in fact, but I'll write it anyways)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

#create the Cross-Validation pipeline
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100)) 

#declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth' : [None, 5, 3, 1]}

#use cross-Validation for the hyperparameters
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

#fit and tune model
clf.fit(X_train, y_train)

#predict the test set of data
y_pred = clf.predict(X_test)

#print metrics
print('R2Score: {}'.format(r2_score(y_test, y_pred)))
print('Mean squared error: {}'.format(mean_squared_error(y_test, y_pred)))

#save the model 
clf2 = joblib.dump(clf, 'rf_regressor.pkl')
 
 
