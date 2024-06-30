Linear regression: 
Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.
Applications of Linear Regression
1.	Predicting body weight based on calorie intake.
2.	Predicting student performance based on the number of hours studied.
3.	Predicting sales based on advertising expenditure.
4.	Predicting house prices based on square footage.
![image](https://github.com/RabbiyaYounas/Linear_regression_model/assets/171420965/284fe5ff-d81c-4b23-b034-9d1b8b1f998e)

Steps for Simple linear regression model 
1) import necessary libraries
   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import seaborn as sns
   # Ensure plots are displayed inline
    %matplotlib inline

2) import dataset
   companies = pd.read_csv('/Users/rabbiyayounas/Documents/DatasetsML/1000_Companies.csv')

3) define dependent and independent variable
#X will contain all columns except the last one.
#y will contain the values from the 5th column (index 4).

X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

4) see dataset to clearly understand
   companies.head()

5) # Encoding categorical data
#Label Encoding: Converts each category to a number.
#One-Hot Encoding: Converts each category to a binary vector (array of 0s and 1s).
# row 3 is names of states. Linear regression model understand numbers only so we converted 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

6) # Avoiding the Dummy Variable Trap
X = X[:, 1:]

7) # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

8) # Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

9) # Predicting the Test set results
#It takes X_test as input and computes predicted values (y_pred) based on the learned relationships between input features and the target variable from the training phase.
y_pred = regressor.predict(X_test)
y_pred

10) # Calculating the Coefficients
#The coefficients (regressor.coef_) indicate the impact of each feature on the predicted outcome (y).
print(regressor.coef_)

11) # Calculating the Intercept
#The intercept (regressor.intercept_) in this context represents the base price of a house when its size (X) is zero.
print(regressor.intercept_)

12) # Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


It ranges from 0 to 1, where:

r2 =1: Indicates that the model perfectly predicts the target variable.

R2=0: Indicates that the model does not explain any of the variability in the target variable around its mean.

R2 <0: Indicates that the model performs worse than simply using the mean of the target variable for prediction.


   
