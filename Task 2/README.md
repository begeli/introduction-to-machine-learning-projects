## Subtask 1: Ordering of Medical Tests

Here we are interested in anticipating the future needs of the patient. You have to predict whether a certain medical test is ordered by a clinician in the remaining stay. This sub-task is a binary classification : 0 means that there will be no further tests of this kind ordered whereas 1 means that at least one is ordered in the remaining stay.

The corresponding columns containing the binary ground truth in train_labels.csv are: LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.

Because there is an imbalance between labels in these sub-tasks we evaluate the performance of a model with the Area Under the Receiver Operating Characteristic Curve, which is a threshold-based metric.

## Subtask 2: Sepsis Prediction

In this sub-task, we are interested in anticipating future life-threatening events. You have to predict whether a patient is likely to have a sepsis event in the remaining stay. This task is also a binary classification : 0 means that no sepsis will occur, 1 otherwise.

The corresponding column containing the binary ground-truth in train_labels.csv is LABEL_Sepsis.

This task is also imbalanced, thus we’ll also evaluate performance using Area Under the Receiver Operating Characteristic Curve.

## Subtask 3: Key Vital Signs Prediction

In this type of sub-task, we are interested in predicting a more general evolution of the patient state. To this effect, here we aim at predicting the mean value of a vital sign in the remaining stay. This is a regression task.

The corresponding columns containing the real-valued ground truth in train_labels.csv are: LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate.

To evaluate the performance of a given model on this sub-task we use R2 Score.

## Methodology

Initially, for every patient we transformed the input data, so, 
that every measurement that could have been taken (including nans) over their 12 hour state was a feature of that patient.
We then eliminated samples with over 95% nans for all of their futures and removed features with over 95% nans for all samples. For the remaining samples and features we applied a variety of data imputation techniques. We decided to settle on KNN imputer because it was faster and empirically provided good or comparable results to other imputers we tried. 
Then we applied recursive feature elimination to get the most relevant features for the models we were training but we couldn't get good results for subtask 2. 
So, we decided to handcraft our features instead of imputing them. We crafted 7 new features per column of the training data (age, mean, median, max, min, std, 0.9th quantile, no. of non-nans per column for a patient). We initially used logistic regression and SVM for the first two tasks but they were slow and they didn't provide good predictions for task 2. So, we decided to use XGBoost library because it was faster and it was capable of handling nan values without preprocessing or imputation. 
We empirically decided to use XGBRFClassifier for the classification tasks. We used Grid Search with 5 fold cross validation (default) to select optimal hyperparameters for the classification. Since we used decision trees, our model overfitted training dataset but it generalized well enough for test dataset too, so, we didn't handle that problem. 
For task 3, we used Lasso regression to reduce complexity of our model and give 0 weights to unnecessary features along with RFE to remove unimportant features. 
We used our handcrafted features then imputed them with KNN imputer because scikit learn can't handle nans. 
We used 20% of our training data for validation to check for overfitting for all 3 tasks.
