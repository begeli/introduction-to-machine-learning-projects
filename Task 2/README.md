# Methodology

Initially, for every patient we transformed the input data, so, 
that every measurement that could have been taken (including nans) over their 12 hour state was a feature of that patient.
We then eliminated samples with over 95% nans for all of their futures and removed features with over 95% nans for all samples.  
For the remaining samples and features we applied a variety of data imputation techniques. We decided to settle on KNN imputer because it was faster and empirically provided good or comparable results to other imputers we tried. 
Then we applied recursive feature elimination to get the most relevant features for the models we were training but we couldn't get good results for subtask 2. 
So, we decided to handcraft our features instead of imputing them. We crafted 7 new features per column of the training data (age, mean, median, max, min, std, 0.9th quantile, no. of non-nans per column for a patient). We initially used logistic regression and SVM for the first two tasks but they were slow and they didn't provide good predictions for task 2. So, we decided to use XGBoost library because it was faster and it was capable of handling nan values without preprocessing or imputation. 
We empirically decided to use XGBRFClassifier for the classification tasks. We used Grid Search with 5 fold cross validation (default) to select optimal hyperparameters for the classification. Since we used decision trees, our model overfitted training dataset but it generalized well enough for test dataset too, so, we didn't handle that problem. 
For task 3, we used Lasso regression to reduce complexity of our model and give 0 weights to unnecessary features along with RFE to remove unimportant features. 
We used our handcrafted features then imputed them with KNN imputer because scikit learn can't handle nans. 
We used 20% of our training data for validation to check for overfitting for all 3 tasks.
