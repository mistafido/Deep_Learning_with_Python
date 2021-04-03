# Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging


logging.basicConfig(filename='data_preprocessing.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#Importing the Dataset
logging.debug('Start of Program')
logging.debug('Initializing the Dataset')
dataset = pd.read_csv('Data.csv')
logging.debug('Dataset Initialized..')
logging.debug(dataset)
logging.debug('Splitting the dataset into X and Y tables')
logging.debug('Creating the X table')
X = dataset.iloc[:, :-1].values
logging.debug('X table created')
logging.debug(X)
logging.debug('Creating the Y table')
y = dataset.iloc[:, -1].values
logging.debug('Y table created')
logging.debug(y)
    
#Taking care of missing data
logging.debug('Taking care of Missing Data in the dataset')
from sklearn.preprocessing import Imputer
logging.debug('Creating the imputer variable from the Imputer class in sklearn.preprocessing package')
logging.debug('Encoding the missing data with mean of that column')
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis =0)
logging.debug('Fitting the imputer variable with the X table')
imputer.fit(X[:, 1:3])
logging.debug('Transforming the imputer variable')
X[:, 1:3] = imputer.transform(X[:, 1:3])
logging.debug('New value of X with imputer variable as missing variable')
logging.debug(X)

#Encoding the categorical data
logging.debug('Encoding categorical Data from string to integer using labelencoder and onehotencoder')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
logging.debug('Creating the LabelEncoder for country variabale')
labelencoder_X = LabelEncoder()
logging.debug('Selecting the  column of variables to encode')
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
logging.debug('Encoding the X variable data column')
X = onehotencoder.fit_transform(X).toarray()
logging.debug('Encoded X variables')
logging.debug(X)
logging.debug('Creating the LabelEncoder for y variabale')
labelencoder_y = LabelEncoder()
logging.debug('Encoding the y variable')
y = labelencoder_y.fit_transform(y)
logging.debug('Encoded y variable')
logging.debug(y)

#Splitting the dataset
logging.debug('Splitting the dataset into training set and test rest to build a neural network on')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
logging.debug('Training set')
logging.debug(X_train)
logging.debug(y_train)
logging.debug('Test set')
logging.debug('X_test')
logging.debug(X_test)
logging.debug('y_test')
logging.debug(y_test)

#Feature Scaling
logging.debug('Scaling the dataset to a standard format using the ')
from sklearn.preprocessing import StandardScaler
logging.debug('Calling the StandardScaler method')
sc_X = StandardScaler()
logging.debug('Fitting the X_train table on the standard scaler')
X_train = sc_X.fit_transform(X_train)
logging.debug('Scaled X_trian Table')
logging.debug(X_train)
logging.debug('Fitting the X_test table on the standard scaler')
X_test = sc_X.fit_transform(X_test)
logging.debug('Scaled X_test Table')
logging.debug(X_test)
