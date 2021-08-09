import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

data = pd.read_csv('dataset/USA_Housing.csv')
data.head()

#to get the basic description of the dataset (mean, variance and standard deviation)
data.describe()
#to get information about the rows and columns and type of the data set
data.info()

#using seaborn to create a jointplot to compare the price of housing and the Area population
sns.jointplot(data=data,x='Price',y='Area Population')

#using seaborn to create a jointplot to compare the price of housing and the avg area Income
sns.jointplot(data=data,x='Price',y='Avg. Area Income')

#using seaborn jointplot to compare the Avg. Area House Age and the Price
sns.jointplot(x='Avg. Area House Age',y='Price',kind='hex',data=data)

#Recreating these relationships across the entire dataset using pairplot
sns.pairplot(data=data)

#from the dataset it can be observed that the Price is most correlated with the Avg. Area Income
#Using searborn's lmplot create a linear model plot of Avg. Area Income vs Price
sns.lmplot(x='Avg. Area Income',y='Price',data=data)

#We have explored the data now lets is split the data into training and testing dataset
#set a variable X to the numerical features of the houses and a variable y to the Price column
data.columns

y = data['Price']

X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

#use model_selection.train_test_split from sklearn to split the data into
#training and testing datasets. set test_size=0.3 and random_state=101

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Training the model
#import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression
#creat an instance of the linearRegression model
lm = LinearRegression()
#Trian/fit the lm on the training data
lm.fit(X_train,y_train)
#print out the coefficients of the model
lm.coef_
#predicting the test data
#use lm.predict() to predict off the X_test set of the data
predictions = lm.predict(X_test)
#using scatterplot of the real test values vs the predicted value
plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')

#Evaluating the model
#Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error
from sklearn import metrics

print('MAE ', metrics.mean_absolute_error(y_test,predictions))
print('MSE ', metrics.mean_squared_error(y_test,predictions))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))

#to check how much variance the models explains
metrics.explained_variance_score(y_test,predictions)

#Residuals

sns.distplot((y_test-predictions),bins=50)

#Conclusions

#recreate the dataframe
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf