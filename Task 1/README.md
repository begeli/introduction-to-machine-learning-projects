# Methodology 

## Task 1a

In this project, we implemented model selection algorithm using cross validation. 
We used cross validation with KFolds cross validator provided by sklearn library with the number of folds (k) set to 10. 
Before we started training our predictors with the folds obtained from KFolds cross validator, 
we set the shuffle parameter to true which shuffled the training dataset provided to us to train better predictors. 
Shuffling probably helped with prediction rate by distributing the data better over the folds and decreasing overfitting. 
For each regularization parameter (outer loop, each regularization parameter per iteration is stored in variable lm), 
we trained 10 predictors (inner loop) where the cross validator split the actual training set into training and validation set where for each predictor these sets are different. 
We trained each predictors (reg.fit(Training points from the cross validator, Corresponding values of the selected training points)) with the training set using ridge regression 
(reg = Ridge(alpha = lm) function), setting the regularization parameter to the value in the outer loop (lm). 
We then predicted the accuracy of each predictor using the validation points given by the cross validator 
(y = reg.predict(Training points selected by the cross validator)). 
We finally calculated the RMSE for the predictor and stored it in an array to calculate the average RMSE of the 10 predictors trained. 
Lowest average RMSE indicates the most accurate reg. parameter.

## Task 1b

In this project, we applied nonlinear feature transformations to the features in our training set. 
Using nonlinear features will help with getting a predictor that generalizes better. We then used used ridge regression to obtain our predictor. 
To get the best regularization parameter, we used cross validation with 10 folds in an interval of 100 equally spaced potential lambdas from 1 to 100. 40 turned out to be the best regularization parameter. 
To judge the accuracy of our weight results, we then used RMSE to evaluate the predictor values with our ground truth labels. 
We considered using lasso penalty because at first we thought reducing the number of features we use would simplify our model and prevent overfitting, however, the error we received from those submissions were higher than the error we received from ridge regression, so, we scrapped that idea.
