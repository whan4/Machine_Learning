'''Use Python to compute the weight vector w = [w0,w1,w2]T for linear regression of N 
input data points (xi,yi) that minimizes the sum of squared error (wTxi –yi)2 . 
The input data are 
x1=(1,3,2)T    y1 =  3
x2=(1,2,2)T    y2 =  1
x3=(1,1,-1)T   y3 =  -1
x4=(1,-1,-3)T  y4 =  -3
'''
import numpy as np
x = np.matrix("1,3,2;1,2,2;1,1,-1;1,-1,-3")
y = np.matrix("3;1;-1;-3")
xt = x.transpose()
from numpy.linalg import inv
w = inv(xt*x)*xt*y
wt = w.T
print(wt)

'''Use the linearRegression module in Python package sklearn to find w for linear regression. '''
import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([[3,2],[2,2],[1,-1],[-1,-3]])
y = np.array([3,1,-1,-3])
model = LinearRegression()
reg = model.fit(x,y)
w_sklearn = np.array([model.intercept_]+list(model.coef_))
print("Weight vector w using scikit-learn:", w_sklearn)