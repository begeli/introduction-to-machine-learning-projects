## Task

The goal of this task is to classify mutations of a human antibody protein into active (1) and inactive (0) based on the provided mutation information. 
Under active mutations the protein retains its original function, and inactive mutation cause the protein to lose its function. 
The mutations differ from each other by 4 amino acids in 4 respective sites. The sites or locations of the mutations are fixed. 
The amino acids at the 4 mutation sites are given as 4-letter combinations, where each letter denotes the amino acid at the corresponding mutation site. 
Amino acids at other places are kept the same and are not provided.

## Methodology 

In this project, we first preprocessed the data by flat mapping the 4 letter combinations to their corresponding label. 
We then encoded each amino acid using One Hot Encoder. In the encoding of each amino acid, there is one number for each possible amino acid and these numbers can be either 1 or 0.
The index corresponding to our current amino acid is 1 and the rest are 0. 
Then we combined the encodings of the 4 amino acids in our 4 letter combination to get our features to train our model. 
In this submission we preferred to use an xgboost classifier because it provided better validation scores than using SVM in most of our trials. 
It was also faster. We used gridsearchCV to find the optimal hyperparameters for the xgboost classifier in our previous submissions. 
We also used 30% of our data set for validation. In this submission, in hopes of training a better model by using less of our data set for validation, 
we used cross validation with 10 folds to get a better estimate or risk. We used the optimal hyperparameters we found in our previous submission with gridsearchCV.
