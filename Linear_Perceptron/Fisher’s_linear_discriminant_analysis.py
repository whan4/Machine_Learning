'''In Fisher’s linear discriminant analysis, 
we search for a vector w such that all data points are well separately after they are projected to the direction defined by w. 
The input data points are 
x1=(4,3)T    y1 =  1
x2=(2,2)T    y2 =  1
x3=(1,1)T    y3 = -1
x4=(-2,-1)T  y4=  -1
(Note that the first element in vector xi is NOT x0=1). 
'''
import numpy as np
x1 = np.matrix("4, 3")
x2 = np.matrix("2, 2")
x3 = np.matrix("1, 1")
x4 = np.matrix("-2, -1")
class_1 = (x1,x2)
class_2 = (x3,x4)
m1 = np.mean(class_1,axis=0)
m2 = np.mean(class_2,axis=0)
Sw = np.dot((x1 - m1).T,(x1 - m1))+np.dot((x2 - m1).T,(x2 - m1))+np.dot((x3 - m2).T,(x3 - m2))+np.dot((x4 - m2).T,(x4 - m2))
print(m1,m2,Sw)
from numpy.linalg import inv
W = inv(Sw)*(m1-m2).T
print(W)
