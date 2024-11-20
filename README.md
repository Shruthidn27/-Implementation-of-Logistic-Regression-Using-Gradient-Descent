# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Preprocess Data: Read the dataset, drop unnecessary columns, and convert categorical variables into numerical codes using .astype('category') and .cat.codes.
2. Define Variables: Split the dataset into features (X) and target variable (Y), and initialize a random parameter vector theta.
3. Implement Functions: Define the sigmoid, loss, gradient_descent, and predict functions for logistic regression.
4. Train Model: Use gradient descent to optimize the parameters theta over a specified number of iterations.
5. Evaluate and Predict: Calculate accuracy of predictions on the training data, and demonstrate predictions with new sample data.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: shruthi D.N
RegisterNumber:  212223240155
*/
import pandas as pd
import numpy as np
data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
X=data1.iloc[:,: -1]
Y=data1["status"]
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
return 1/(1+np.exp(-z))
def loss(theta,X,y):
h=sigmoid(X.dot(theta))
return -np.sum(y*np.log(h)+ (1-y) * np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
m=len(y)
for i in range(num_iterations):
h=sigmoid(X.dot(theta))
gradient=X.T.dot(h-y)/m
theta-=alpha*gradient
return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
h=sigmoid(X.dot(theta))
y_pred=np.where(h>=0.5 , 1,0)
return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```

## Output:
## ACCURACY:

![image](https://github.com/user-attachments/assets/daa55a45-7615-4d00-8641-c9d4e81b8dab)




## PREDICTED:

![image](https://github.com/user-attachments/assets/f3b4ad0f-e6a2-41ba-b5bc-0746ae51fbd4)

## ACTUAL:

![image](https://github.com/user-attachments/assets/869f4af6-1726-4429-bdf9-1ffd2489b7b5)

## PREDICTED RESULT:

![image](https://github.com/user-attachments/assets/c273b440-7858-4ac6-8045-6707a01534a5)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

