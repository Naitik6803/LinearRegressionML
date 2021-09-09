import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

dia=datasets.load_diabetes()


#print('DataSetName'.keys()) ------> to show all keys in that particular dataset
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])



dia_X= dia.data
dia_X_train=dia_X[:-50]
dia_X_test=dia_X[-50:]

dia_Y_train=dia.target[:-50]
dia_Y_test=dia.target[-50:]

model=linear_model.LinearRegression()  #creating model for diabetes
model.fit(dia_X_train,dia_Y_train) #

predicition =model.predict(dia_X_test)

print("Mean squared Error :",mean_squared_error(dia_Y_test,predicition))
print("Weights :",model.coef_)
print("Intercepts :",model.intercept_)

#Mean squared Error : 1789.5698810318208
#lesser MSE due to more quality features