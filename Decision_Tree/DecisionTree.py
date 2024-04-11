'''
An immunotherapy data set contains information about wart treatment results using immunotherapy. 
The sheet “training data set” contains information of 80 training data points and the sheet “test data set” contains information of 10 test data points. 
Use the class DecisionTreeClassifier in the sklearn Python package to build a decision tree based on the 80 training data points. '''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_excel("Immunotherapy.xlsx", sheet_name="Training data set")
test_data = pd.read_excel("Immunotherapy.xlsx", sheet_name="Test data set")

X_train = train_data.iloc[:, :-1]  
y_train = train_data.iloc[:, -1]   

X_test = test_data.iloc[:, :-1]   

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(predictions)
