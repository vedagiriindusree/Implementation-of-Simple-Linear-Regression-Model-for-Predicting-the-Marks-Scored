# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vedagiri Indu Sree
RegisterNumber:  212223230236
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()

#segregating data to variables
X = df.iloc[:,:-1].values
print(X)

Y=df.iloc[:,1].values
print(Y)

#splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print(Y_pred)

#display actual values
print(Y_test)

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

#Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_test,Y_pred,color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

## Displaying the content in datafield

## head:

![image](https://github.com/user-attachments/assets/0f932e2e-c9a7-43fb-9532-5fd97c1f61c8)

## tail:
![image](https://github.com/user-attachments/assets/66e95651-72c5-4ae0-9ea7-d30a6ec6101f)

## Segregating data to variables
![image](https://github.com/user-attachments/assets/dfe6152d-26e0-4286-b68c-4eec6e36fa24)
![image](https://github.com/user-attachments/assets/d1033c8e-38d1-4078-88c8-0a2e9b42111e)


## Displaying predicted values
![image](https://github.com/user-attachments/assets/07e56b3a-8941-42db-8b4f-615a44b80b64)

## Displaying actual values
![image](https://github.com/user-attachments/assets/87706d7f-bd46-44a7-8e92-7e8f918f5901)

## MSE MAE RMSE
![image](https://github.com/user-attachments/assets/2f6235e4-5dd0-4512-bb58-aaaa0f0e7157)

## Graph plot for training data
![image](https://github.com/user-attachments/assets/8c7c10f4-512d-45f9-a0e2-3098b9f939e7)

## Graph plot for test data
![image](https://github.com/user-attachments/assets/8509c5d0-3309-42f1-9a69-52c9c609da5d)

## Result
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
