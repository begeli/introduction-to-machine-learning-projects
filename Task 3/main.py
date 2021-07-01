import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost
####################################### load data #####################################

print("initializing")
## reading training data from the CSV file
training_data_df = pd.read_csv (r'train.csv')  #read cvs

#extract the protein sequence from the data
training_sequence = list(training_data_df['Sequence'])

# extract the labels form the data 
training_labels = training_data_df['Active'].to_numpy()
#print(np.mean(training_labels))

# split the string containing the proteins to seperate characters 
# to be used for encoding the protein at each location
train_seq= []
for sequence in training_sequence:
    train_seq.append(list(sequence))

# encode the proteins with one hot encoding and convert to numpy array
enc = OneHotEncoder(handle_unknown='ignore')
train_encoded = enc.fit_transform(train_seq).toarray()



# same things repeated for the test data
test_data_df = pd.read_csv (r'test.csv')  #read cvs

test_sequence = list(test_data_df['Sequence'])
test_seq= []

for sequence in test_sequence:
    test_seq.append(list(sequence))

test_data = enc.transform(test_seq).toarray()

##### this part was used for model selection and is now commented out ##########

#training_data, validation_data, train_y, val_y = train_test_split(train_encoded, training_labels, test_size=0.2, random_state=13)            #SPLITTIN DATASETS!!!!!

# ## Grid Search on hyper-parameters
# n_estimator_list = [500, 550, 600]
# #subsample_list = [0.4, 0.6]
# #colsample_bynode_list = [0.4, 0.5, 0.6]
# max_depth_list = [50, 100, 200]
# reg_alpha_list = [1.0 ,5.0, 10.0]
# #scale_pos_weight_list = [0.9, 1]
# #param_dict = {'n_estimators':n_estimator_list, 'subsample':subsample_list, 'colsample_bynode':colsample_bynode_list, 'max_depth':max_depth_list,'reg_alpha':reg_alpha_list, 'scale_pos_weight':scale_pos_weight_list}
# param_dict = {'n_estimators':n_estimator_list, 'max_depth':max_depth_list, 'reg_alpha':reg_alpha_list}
# #Using XGBClassifier boost model to classify
# model_xgb = xgboost.XGBClassifier(objective='binary:logistic',eval_metric = 'auc', use_label_encoder=False, random_state=42 )
# grid_search = GridSearchCV(model_xgb, param_dict, refit=True,cv=3)
# grid_search.fit(training_data, train_y)
# print(grid_search.best_params_)
# print('Training score')
# print(f1_score(train_y, grid_search.predict(training_data)))
# print('Validation score')
# print(f1_score(val_y, grid_search.predict(validation_data)))
# test_label_predictions = grid_search.predict(test_data)


### checking the 10-fold cross validation scores of the selected model to see performance
sum=0
kf = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in kf.split(train_encoded,training_labels):
    X_train, X_val = train_encoded[train_index], train_encoded[test_index]
    y_train, y_val = training_labels[train_index], training_labels[test_index]
    #Using XGBClassifier boost model to classify
    model_xgb = xgboost.XGBClassifier(n_estimators=500, max_depth=50, reg_alpha=1.0,colsample_bynode=0.5, objective='binary:logistic',eval_metric = 'auc', use_label_encoder=False, random_state=42,scale_pos_weight=5, subsample=0.8)
    model_xgb.fit(X_train, y_train)
    print('Training score')
    print(f1_score(y_train, model_xgb.predict(X_train)))
    print('Validation score')
    print(f1_score(y_val, model_xgb.predict(X_val)))
    sum = sum + f1_score(y_val, model_xgb.predict(X_val))
print(sum/10.0)

# selected model for submission 
model_final = xgboost.XGBClassifier(n_estimators=500, max_depth=50, reg_alpha=1.0, colsample_bynode=0.5,
                                  objective='binary:logistic', eval_metric='auc', use_label_encoder=False,
                                  random_state=42, scale_pos_weight=5, subsample=0.8)

# train on the whole training set after model selection 
model_final.fit(train_encoded, training_labels)
test_label_predictions = model_final.predict(test_data)

## write the weights to submission file
df = pd.DataFrame(test_label_predictions)
df.to_csv('submission.csv', index=False, header=None) #write CSV
