# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Aadithya.R 
RegisterNumber:  212223240001
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse) 
```

## Output:
### df.head()
![1st](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/77163267-9ac2-467b-803d-cbbb5d8682ae)

### df.tail()
![2](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/96c0a72b-a114-45ba-a43a-59e33a4b718e)

### Array value of X
![3](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/bf70d1b6-dddb-492a-b1ae-5788a4050c7d)

### Array value of Y
![4](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/b725c8f6-a33d-4cd3-97c9-2f34b96115da)

### Values of y prediction
![5](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/2588d8af-ccd8-464c-adaf-eb18b67ba69f)

### Array values of Y test
![6](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/41e9e8e8-170c-489a-995a-1c79e3556202)

### Training set graph
![7](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/53762525-6ea8-4f33-9d0e-522fc467790e)

### Test set graph
![8](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/95c68297-0377-45ec-a21e-84f801ba48d2)

### Values of MSE,MAE and RMSE
![9](https://github.com/Aadithya2201/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145917810/da31d005-1efa-4c44-b33b-6908d4be9350)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
