## Task

In this task, we make decisions on food taste similarity based on images and human judgements. 
We are provided with a dataset of images of 10.000 dishes.
We are also provided with a set of triplets (A, B, C) representing human annotations: 
the human annotator judged that the taste of dish A is more similar to the taste of dish B than to the taste of dish C.
Our task is to predict for unseen triplets (A, B, C) whether dish A is more similar in taste to B or C.

## Methodology 

Our project consists of 4 parts: The net.py file which is the network architecture we used, loader.py file which is the file we used to load normalized images, 
train.py where we train our neural network and main.py where we split our data set into train/validation sets and classify images with our trained neural network. 
Our neural network implements the deep ranking architecture. It consists of 3 parallel, embedded networks (a, p, n) each of which consists of 2 parts. 
1st one is the backbone which is a pretrained resnet18 model. 2nd one is a fully connected linear layer which outputs an embedding with 1024 dimensions. 
This output is practically a point where the image is mapped to. Embedding network "a" is the network for query, embedding network "p" is the network for positive 
image and the embedding network "n" is the network for negative image. In our network, if the image a is closer to image p then the classification class is 1 otherwise it is 0. 
Loader.py loads the triplet images from the file specified and it normalizes them according the precalculated means and variances. 
It also resizes the images to the minimum x and y dimensions to ensure all images are of the same dimensions. For training we use SGD with 1 epoch. 
Each batch size is 120 samples. We use 1 epoch to prevent overfitting. We use TripletMarginLoss as our loss function. 
The margin for validation is 0 as we want to evaluate the accuracy of our classification. In our main.py file we first split the data set into training and 
validation sets but splitting the training triplets. We train our model in GPU. We classify the test set calculating the Euclidean distance between a and p & a and n. 
If p is closer to a than n we classify 1. We tuned all parameters based on our validation score. We used Google Colab to have access to GPUs with 
larger memories which enabled us to train more accurate models because of larger batch sizes.
