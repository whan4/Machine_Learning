'''An immunotherapy data set contains information about wart treatment results using immunotherapy. The sheet “training data set” contains information of 80 training data points and the sheet “test data set” contains information of 10 test data points. 
Use the class SVM in the sklearn package in Python to train a svm classifier with a linear kernel using the 80 training data points.
'''
import pandas as pd
from sklearn.svm import SVC

train_data=pd.read_excel("~/Immunotherapy.xlsx", sheet_name="Training data set")
test_data=pd.read_excel("~/Immunotherapy.xlsx", sheet_name="Test data set")

x_train=train_data.iloc[:, :-1]
y_train=train_data.iloc[:, -1]

x_test=test_data.iloc[:,:-1]

svm_classifier=SVC(kernel="linear")
svm_classifier.fit(x_train,y_train)
y_pred=svm_classifier.predict(x_test)
print("Predicted results on the test data:")
print(y_pred)

