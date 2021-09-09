#Basic linear regression model (based on diabetes dataset in sklearn)
#It's just basic two independent variable plot
#Y= a0 +a1*X1     form
#so accuracy will be less of this model couz only 2 parameters are there
# Lesser the number of quality features, lesser the accuracy ^_^ (lesser the mean squared error)





import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error



dia=datasets.load_diabetes()


#print('DataSetName'.keys()) ------> to show all keys in that particular dataset
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])



dia_X= dia.data[:,np.newaxis,2]
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


#plotting graph of given model
plt.scatter(dia_X_test,dia_Y_test) #plotting data points
plt.plot(dia_X_test,predicition)      #plotting predicted line
plt.show()


#Mean squared Error : 3471.923196056966
#higher MSE due to less features
