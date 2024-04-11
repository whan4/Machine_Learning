'''Use the training data in the immunotherapy data set and the sklearn package in Python to train a logistic regression model for predicting “Result_of_treatment.” 
The input data of the model includes the columns “age”, “number_of_warts”, and “type.” 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_excel("~/Immunotherapy.xlsx", sheet_name="Training data set")

X = data[["age", "Number_of_Warts", "Type"]]  
y = data["Result_of_Treatment"]  

model = LogisticRegression()
model.fit(X, y)

weights = model.coef_
intercept = model.intercept_

print("Coefficients (Weights):")
for feature, weight in zip(X.columns, weights[0]):
    print(f"{feature}: {weight}")

print(f"Intercept: {intercept[0]}")