# Introduction to Machine Learning Projects
Term projects for ETH Zurich Introduction to Machine Learning course. Projects were about practicing supervised learning tasks. 

## Project topics are as follow:

**Task 1a:** Use 10-fold cross validation to determine the best regularization parameters for linear regression task.

**Task 1b:** Given an input vector **x**, predict a value **y** as a linear function of a set of feature transformations.

**Task 2:** This task was primarily about preprocessing data to handle missing values. We were expected to predict the evolution of hospital patients' states and needs during their stay in the Intensive Care Unit (ICU).
* **Subtask 1: (Ordering of Medical Tests)** Predict whether a certain medical test is ordered by a clinician in the remaining stay. **(Binary Classification task)**
* **Subtask 2: (Sepsis Prediction)** Predict whether a patient is likely to have a sepsis event in the remaining stay. **(Binary Classification task)**
* **Subtask 3: (Key Vital Signs Predictions)** Predict the mean value of a vital sign in the remaining stay to predict a more general evolution of the patient state. **(Regression task)**

**Task 3:** Classify mutations of a human antibody protein into active (1) and inactive (0) based on the provided mutation information.

**Task 4:** Make decisions on food taste similarity based on images and human judgements.
* Implemented Deep Ranking architecture and used pre-trained resnet18 model as backbone. The inputs are triplets of images where we try to determine which of the 2nd and 3rd images are closest to the 1st image. Our architecture consists of 3 parallel networks (a, p, n) which are query, positive and negative networks respectively. Our networks performs a binary classification where we output 1 if 2nd image in the triplet is closer to 1st image, 0 otherwise.

## Results

**Task 1b:** Achieved an RMSE of 2.064

**Task 2:** 
* **Subtask 1:** Average AUROC for all the labels classified 0.745 
* **Subtask 2:** AUROC 0.692
* **Subtask 3:** Average R2 score of 0.752 for all the output labels

**Task 3:** Achieved an F1 score of 0.902

**Task 4:** Classified 70.2% of all triplets correctly

## Contributors

* Bartu Soyuer
* Gizem Yüce
* Berke Egeli
