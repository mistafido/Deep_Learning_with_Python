import logging
logging.basicConfig(filename='ann__predict_heart_attack.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of Program')
logging.debug('Importing the Libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
logging.debug('Importing the Dataset')
dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values
logging.debug('Dataset imported')
logging.debug(dataset)
logging.debug('Splitting the dataset into Training set and Test set')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
logging.debug('Training set')
logging.debug(x_train)
logging.debug(y_train)
logging.debug('Test set')
logging.debug(x_test)
logging.debug(y_test)
logging.debug('Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
logging.debug('X_train scaled feature')
logging.debug(x_train)
logging.debug('x_test scaled feature')
logging.debug(x_test)
logging.debug('Building the Artificial Neural Network')
logging.debug('Importing Keras Libraries and Packages')
import keras
from keras.models import Sequential
from keras.layers import Dense
logging.debug('Initializing the Artificial Neural Network')
classifier = Sequential()
logging.debug('Adding the hidden layer and the first hidden layer')
classifier.add(Dense( 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
logging.debug('Adding the second hidden layer')
classifier.add(Dense( 10, kernel_initializer = 'uniform', activation = 'relu' ))
logging.debug('Adding the third hidden layer')
classifier.add(Dense( 10, kernel_initializer = 'uniform', activation = 'relu' ))
logging.debug('Adding the output layer')
classifier.add(Dense( 1, kernel_initializer = 'uniform', activation = 'sigmoid' ))
logging.debug('Compiling the Artificial Neural Network')
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
logging.debug('Fitting the Artificial Neural Network to the trainging set')
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)
logging.debug('Predicting the Test set results')
y_pred = classifier.predict(x_test)
logging.debug('Predicted results')
logging.debug(y_pred)
y_pred = (y_pred > 0.85)
logging.debug('Making the confusion matrix')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
logging.debug('Predicting a single observation')
"""
age:
sex:
cp:
trtbps:
chol:
fbs:
restecg:
thalachh:
exng:
oldpeak:
slp:
caa: 
thall:
"""
new_prediction = classifier.predict(sc.transform(np.array([[64, 1, 0, 185, 351, 1, 2, 240, 1, 1.8, 2, 0, 12]])))
logging.debug('Single observation prediction')
logging.debug(new_prediction)
new_prediction = (new_prediction > 0.85)

