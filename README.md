# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
step1:start
<br>step-2:Import the standard Libraries. 
<br>step-3:Set variables for assigning dataset values. 
<br>step-4:Import linear regression from sklearn. 
<br>step-5:Assign the points for representing in the graph. 
<br>step-6:Predict the regression for marks by using the representation of the graph. 
<br>step-7:stop
## Program:
```
Program to implement the simple linear regression model
for predicting the marks scored.
Developed by: VIKAASH K S
RegisterNumber: 212223240179
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
```
```
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
![df_head()](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/0b7d642e-7301-4319-aafd-a9a4f648b04b)

### df.tail()
![df tail()](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/68fdaf4f-3368-4779-ac98-6541956060f6)

### Array value of X
![array value](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/5a783fe9-924d-48b4-8d15-b636ba4135d1)

### Array value of Y
![array value of y](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/998a4174-131b-498b-a882-fcd39a015aa3)

### Values of y prediction
![value of y prediction](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/d22c7fe9-036e-45c0-86cc-b07a64b06996)
### Array values of Y test
![array values of y test](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/fbe00f6c-0a90-459f-8982-4e5b7e979fdb)

### Training set graph
![training set graph](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/688388ba-4567-424c-b641-c7296f4697f2)

### Test set graph
![test set graph 1](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/548f2c41-76fa-4032-902d-4bf7b0093366)

### Values of MSE,MAE and RMSE
![values of mse,mae and rmse](https://github.com/rohithprem18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/146315115/5de3982e-704d-4750-9802-e398748bf131)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
