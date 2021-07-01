import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

## reading training data from teh CSV file
training_data = pd.read_csv (r'train.csv', dtype= np.float64, float_precision='high')  #read cvs
td = training_data.to_numpy()  #convert the data frame to numpy array
train_y= td[:,1]
train_X= td[:,2:]

## creating the feature matrix phi
linear = train_X
quadratic = np.square(train_X)
exponential = np.exp(train_X)
cosine = np.cos(train_X)
constant = np.ones((train_X.shape[0],1), dtype=np.float64)
feature_matrix = np.concatenate((linear, quadratic, exponential, cosine, constant), axis=1)

## ridge regression
#lmbd = 4.75
#ridge_reg = Ridge(alpha = lmbd)
#ridge_reg.fit(feature_matrix, train_y)
#y_predicted = ridge_reg.predict(feature_matrix)
#RMSE = mean_squared_error(y_predicted,train_y)**0.5
#print(RMSE)
#weights = ridge_reg.coef_
#print(weights)

## Lasso regression
#lasso_reg = Lasso(alpha =  0.0048)
#lasso_reg.fit(feature_matrix, train_y)
#y_predicted = lasso_reg.predict(feature_matrix)
#RMSE = mean_squared_error(y_predicted,train_y)**0.5
#print(RMSE)
#weights = lasso_reg.coef_

## Lasso with CV
# reg = LassoCV(cv=10, fit_intercept=False, random_state=0, alphas=1e-2*np.linspace(1,10, num=100)).fit(feature_matrix, train_y)
# print(reg.alpha_)
# weights = reg.coef_
# print(weights)
# y_predicted = reg.predict(feature_matrix)
# RMSE = mean_squared_error(y_predicted,train_y)**0.5
# print(RMSE)


# Ridge regression with CV
reg = RidgeCV(cv=10, alphas=np.linspace(1,100, num=100), fit_intercept=False ).fit(feature_matrix, train_y)
weights = reg.coef_
y_predicted = reg.predict(feature_matrix)
#y = np.matmul(feature_matrix, weights)
RMSE = mean_squared_error(y_predicted,train_y)**0.5

print('optimal alpha')
print(reg.alpha_) 
print('optimal weights')
print(weights)
print('RMSE')
print(RMSE)

## Linear Regression with no regularization
#lin_reg = LinearRegression().fit(feature_matrix,train_y)
#weights = lin_reg.coef_
#y_predicted = lin_reg.predict(feature_matrix)
#RMSE = mean_squared_error(y_predicted,train_y)**0.5

## write the weights to submission file
df = pd.DataFrame(weights)
df.to_csv('submission.csv', index=False, header=False) #write CSV

