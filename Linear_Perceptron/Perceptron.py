'''A heart disease data set contains the medical information of 100 patients of two classes. 
The first row contains column names, and other rows contain the information of the patients. 
The last column “num” contains the classes of the patients. 
Use the perceptron module in Python.
'''
import numpy as np
data = np.genfromtxt("~/heart_disease_data.txt", delimiter=",", skip_header=1)
print(data)
X = data[:, :-1]
y = data[:,-1]
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X,y)
w_perceptron = perceptron.coef_
w0_perceptron = perceptron.intercept_
print(w_perceptron,w0_perceptron)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X,y)
w_lda = lda.coef_
w0_lda = lda.intercept_
print(w_lda, w0_lda)

# print(X,y)
# age=data[:,0]
# Sex=data[:,1]
# cp=data[:,2]
# trestbps=data[:,3]
# chol=data[:,4]
# fbs=data[:,5]
# restecg=data[:,6]
# thalach=data[:,7]
# exang=data[:,8]
# oldpeak=data[:,9]
# slope=data[:,10]
# thal=data[:,11]
# num=data[:,13]



