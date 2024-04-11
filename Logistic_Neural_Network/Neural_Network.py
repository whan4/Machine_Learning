'''Use the keras package in Python to build a neural network with two hidden layers and an output layer, in which tanh is the activation function. 
You can decide the number of neurons in each layer. 
Use the training data set to train the network so that it can predict treatment results using the information of sex, age, time, number of warts, type, area, induration diameter.
'''
import keras.models as km 
import keras.layers as kl 
import pandas as pd
data = pd.read_excel("~/Immunotherapy.xlsx", sheet_name="Training data set")
X = data[["sex","age", "Time","Number_of_Warts", "Type","Area","induration_diameter",]]  
Y = data["Result_of_Treatment"]  
model = km.Sequential() 
model.add(kl.Dense(48, input_dim=7, use_bias=True, bias_initializer='ones', activation='tanh')) 
model.add(kl.Dense(4, use_bias=True, bias_initializer='ones', activation='tanh')) 
model.add(kl.Dense(1, use_bias=True, bias_initializer='ones', activation='tanh')) 
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) 
model.fit(X, Y, epochs=1000) 
z=model.predict(X)
print(z)                            