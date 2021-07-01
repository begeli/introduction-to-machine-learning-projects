import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import KFold

training_data = pd.read_csv (r'train.csv')  #read cvs
td = training_data.to_numpy()  #convert the data frame to numpy array
train_y= td[:,0] 
train_X= td[:,1:]

np.random.seed(7) #fix random seed 

lmbd = [0.1, 1, 10, 100, 200] #given lambda values
K=10
kf10 = KFold(n_splits=K, shuffle=True)

RMSE_lm=[] #holds the average RMSE value for each lambda
for lm in lmbd:
    RMSE=[] #holds the RMSE value for different folds
    for train_index, test_index in kf10.split(train_X): #loop over 10 folds
        reg = Ridge(alpha = lm)
        reg.fit(np.take(train_X,train_index,axis=0),np.take(train_y,train_index))
        y = reg.predict(np.take(train_X,test_index,axis=0))
        RMSE.append(mean_squared_error(y, np.take(train_y,test_index))**0.5) 

    RMSE_avg = np.mean(RMSE) #average of the RMSE values for different folds for the given lambda value
    RMSE_lm.append(RMSE_avg)

print(RMSE_lm)
df = pd.DataFrame(RMSE_lm)
df.to_csv('submission.csv', index=False, header=False) #write CSV
