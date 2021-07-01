import pandas as pd
import numpy as np
import xgboost

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics
from sklearn import linear_model

########################### functions for feature extraction #########################################

#Function defined to generate the hand-crafted features defined with respect to the 12 hour measurements of each patient
def patient(patient_rows):
    is_feature_nan = np.sum(np.isnan(patient_rows) + 0, axis=0)  # number of nan values on each fatures for each patient
    tests_done = 12 - is_feature_nan
    mean_features = np.nanmean(patient_rows, axis=0)
    max_features = np.nanmax(patient_rows, axis=0)
    min_features = np.nanmin(patient_rows, axis=0)
    median_features = np.nanmedian(patient_rows, axis=0)
    std_features = np.nanstd(patient_rows, axis=0)
    quantile_features = np.nanquantile(patient_rows, 0.9, axis=0)

    patient_features = np.concatenate((mean_features, max_features, min_features, median_features, std_features, tests_done, quantile_features))
    #For each patient,the means, maximum, minimum, median, standarad dev and 0.9th quantile of each feature (along with age) constitute a total of 239 'new' features
    return patient_features

#Function defined to construct the whole dataset from the new features that were hand crafted
def extract_features(td):
    pid = td[:,0]
    time = td[:,1]
    age = td[:,2]
    age = age[0::12]

    raw_data = td[:,3:]
    n_rows = raw_data.shape[0]    #number of patients*12
    n_columns = raw_data.shape[1]  #number of features
    reshaped_data = raw_data.reshape((12, int(n_columns*n_rows/12)))    #reshape the data

    training_data_final = np.zeros( (int(td.shape[0]/12),238) )

    for i in range(int(td.shape[0]/12)):
        patient_features = patient(raw_data[12*i:12*i+12,::])
        training_data_final[i,:] = patient_features

    training_data = training_data_final
    training_data = np.concatenate((np.expand_dims(age, axis=1), training_data), axis=1)

    return training_data

####################################### load data #####################################

print("initializing")
## reading training data from teh CSV file
training_data = pd.read_csv (r'train_features.csv', dtype= np.float32, float_precision='high')  #read cvs
td = training_data.to_numpy()  #convert the data frame to numpy array


## reading test data from the CSV file
test_data = pd.read_csv (r'test_features.csv', dtype= np.float32, float_precision='high')  #read cvs
td_test = test_data.to_numpy()  #convert the data frame to numpy array

#Separating the test data pid, time and raw data columns
pid_test = td_test[:,0]
time = td_test[:,1]
test_data = td_test[:,2:]
pid_test_test = pid_test[0::12]

## read labels from the csv
training_labels = pd.read_csv (r'train_labels.csv', dtype= np.float32, float_precision='high')  #read cvs
td_labels = training_labels.to_numpy()  #convert the data frame to numpy array$


 #Generating the test and training sets with the new features and saving them for future use
training_data = extract_features(td)
test_data = extract_features(td_test)

df = pd.DataFrame(training_data)   
df.to_csv('feat_ex_train.csv', index=False) #write CSV
df = pd.DataFrame(training_data)
df.to_csv('feat_ex_test.csv', index=False) #write CSV


#Generating a validation set to monitor overfitting and estimate private/public dataset performance
training_data, validation_data, train_y, val_y = train_test_split(training_data, td_labels, test_size=0.2, random_state=42)            #SPLITTIN DATASETS!!!!!

#parse the labels for different tasks
pid_2_train = train_y[:,0]
task_1_labels_train = train_y[:,1:11]
task_2_label_train = train_y[:,11]
task_3_y_train = train_y[:,12::]

pid_2_val = val_y[:,0]
task_1_labels_val = val_y[:,1:11]
task_2_label_val = val_y[:,11]
task_3_y_val = val_y[:,12::]

print('feature selection completed')

# KNN Imputer is used to impute the nan values in the collected data
imp = KNNImputer(n_neighbors=50)      #To handle the remanining NaNs in the dataset, use KNN imputation to fill in the data
imp.fit(training_data)

train_X_3 = imp.transform(training_data)
test_X_3 = imp.transform(test_data)
val_X_3 = imp.transform(validation_data)

df = pd.DataFrame(train_X_3)
df.to_csv('knn_imp_train_new_features.csv', index=False, header=None) #write CSV

df = pd.DataFrame(val_X_3)
df.to_csv('knn_imp_val_new_features.csv', index=False, header=None) #write CSV

df = pd.DataFrame(test_X_3)
df.to_csv('knn_imp_test_new_features.csv', index=False, header=None) #write CSV
 
print('imputation completed')

# #
# # # #TASK1

train_score1 = np.zeros(task_1_labels_train.shape[1])
task1_score = np.zeros(task_1_labels_train.shape[1])
predicted_labels_task1 = np.zeros((test_data.shape[0],task_1_labels_val.shape[1]))

#The following lists were originally used to find the optimal hyperparameters for our XGBClassifier model via GridSearchCV,
#however, the different value in these lists have been reduced to only the optimal values found for the sake of submitting a
#code with lower run time and for your convenience
n_estimator_list = [500]
subsample_list = [0.4]
colsample_bynode_list = [0.3]
max_depth_list = [15]
reg_alpha_list = [100]
param_dict = {'n_estimators': n_estimator_list, 'subsample': subsample_list, 'colsample_bynode': colsample_bynode_list,
              'max_depth': max_depth_list, }

for i in range(task_1_labels_val.shape[1]): #Train separately for each label
    #Use XGBClassifier Boost model to calssify
    model_task_1 = xgboost.XGBRFClassifier(objective='binary:logistic',eval_metric = 'auc', use_label_encoder=False, random_state=42)
    grid_search = GridSearchCV(model_task_1, param_dict, refit=True)

    # fit the model using the training data
    grid_search.fit(training_data,task_1_labels_train[:,i])
    print('Task1 best parameters:',grid_search.best_params_)

    #make predictions with the trained model
    predicted_labels_task1[:,i] = grid_search.predict_proba(test_data)[:, 1]
    train_predict = grid_search.predict_proba(training_data)[:, 1]
    val_predict = grid_search.predict_proba(validation_data)[:, 1]

    #Score on training set with respect to ROC_AUC Metric
    train_score1[i] = metrics.roc_auc_score(task_1_labels_train[:,i], train_predict)

    #Score on the validation set with respect to the ROC_AUC metric
    task1_score[i] = metrics.roc_auc_score(task_1_labels_val[:,i], val_predict)

    print(train_score1[i])
    print(task1_score[i])

np.save('predicted_labels_task1.npy', predicted_labels_task1)


#TASK2

'''
The following lists were originally used to find the optimal hyperparameters for our XGBClassifier model via GridSearchCV,
however, the different value in these lists have been reduced to only the optimal values found for the sake of submitting a
code with lower run time and for your convenience
'''

n_estimator_list = [500]
subsample_list = [0.5]
colsample_bynode_list = [0.3]
max_depth_list = [20]
reg_alpha_list = [50]
param_dict = {'n_estimators':n_estimator_list, 'subsample':subsample_list, 'colsample_bynode':colsample_bynode_list, 'max_depth':max_depth_list,'reg_alpha':reg_alpha_list}

#Using XGBClassifier boost model to classify
model_task_2 = xgboost.XGBClassifier(objective='binary:logistic',eval_metric = 'auc', use_label_encoder=False, random_state=42 )
grid_search = GridSearchCV(model_task_2, param_dict, refit=True,cv=3)
grid_search.fit(training_data,task_2_label_train)
print('Task2 best parameters:',grid_search.best_params_)
print('Grid search completed')

train_predict = grid_search.predict_proba(training_data)[:, 1]
val_predict = grid_search.predict_proba(validation_data)[:, 1]

#Score on training set with respect to ROC_AUC Metric
train_score2 =  metrics.roc_auc_score(task_2_label_train, train_predict)

#Score on validation set with respect to ROC_AUC Metric
task2_score = metrics.roc_auc_score(task_2_label_val, val_predict)

predicted_labels_task2 = grid_search.predict_proba(test_data)[:, 1]  # predictions on the test set

print(train_score2)
print(task2_score)

np.save('predicted_labels_task2.npy', predicted_labels_task2)

# # #TASK3

#Using LassoCV Regressor on the KNN imputed data
clf = linear_model.MultiTaskLassoCV()  # linear regression with lasso regularization and regularization parameter is chosen with CV
clf.fit(train_X_3, task_3_y_train)

#score_train_3 = clf.score(val_X_3,task_3_y_val)
score_sub_3 = np.zeros(task_3_y_val.shape[1])
score_tr_3 = np.zeros(task_3_y_val.shape[1])
for i in range(task_3_y_val.shape[1]):
    # Score on training set with respect to R^2 score
    score_sub_3[i] = 0.5 + 0.5 * np.maximum(0, metrics.r2_score(task_3_y_val[:,i], clf.predict(val_X_3)[:,i]))

    # Score on training set with respect to R^2 score
    score_tr_3[i] = 0.5 + 0.5 * np.maximum(0, metrics.r2_score(task_3_y_train[:,i], clf.predict(train_X_3)[:,i]))

prediction_task_3 = clf.predict(test_X_3)
print("Task 3 Completed")
print(score_tr_3)
print(score_sub_3)
np.save('prediction_task_3_final.npy',prediction_task_3)


######### write submission file ########

#Define the labels for the header of the submission
PID = ['pid']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

SEPSIS = ['LABEL_Sepsis']

labels = np.concatenate((PID, TESTS, SEPSIS,VITALS))

## write csv
np.save('predicted_labels_task1.npy',predicted_labels_task1)
np.save('predicted_labels_task2.npy',predicted_labels_task2)
np.save('prediction_task_3.npy',prediction_task_3)

submission = np.append(np.expand_dims(pid_test_test,axis=1),np.append(np.append(predicted_labels_task1, np.expand_dims(predicted_labels_task2,axis=1),axis=1), prediction_task_3, axis=1), axis=1)

# write the weights to submission file
df = pd.DataFrame(submission)
df.to_csv('submission_final.csv', index=False, header=labels) #write CSV

# suppose df is a pandas dataframe containing the result
df.to_csv('prediction_final.zip', header=labels, index=False, float_format='%.3f', compression='zip')