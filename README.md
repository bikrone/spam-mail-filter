Spam Mail Filter
===================


This is a simple Spam Mail Filter program written in Python, using Naive Bayes Classification.

----------

###Dataset
Currently the program uses the [Enron-Spam dataset](http://www.aueb.gr/users/ion/data/enron-spam/) (Enron1 & Enron2 as the training set, Enron3 as the test set, you can download more change it later).

Download Enron, Enron2 and Enron3 and extract them into folder ```dataset/enron-spam``` (you can see or change the directories of Trained and Test folder by changing this code)
```python
TRAIN_HAM_FOLDER = ['dataset/enron-spam/enron1/ham', 'dataset/enron-spam/enron2/ham']
TRAIN_SPAM_FOLDER = ['dataset/enron-spam/enron1/spam', 'dataset/enron-spam/enron2/spam']

TEST_HAM_FOLDER = ['dataset/enron-spam/enron2/ham']
TEST_SPAM_FOLDER = ['dataset/enron-spam/enron2/spam']
```

###Algorithm
For each document M, the probability of spam given the message is:
> ![probability of spam](https://latex.codecogs.com/gif.latex?p%28c_s%7CM%29%20=%20%5Cfrac%7Bp%28c_s%29%5Ccdot%20p%28M%20%7C%20c_s%29%7D%7Bp%28c_s%29%5Ccdot%20p%28M%20%7C%20c_s%29%20&plus;%20p%28c_h%29%5Ccdot%20p%28M%20%7C%20c_h%29%7D)

With ![](https://latex.codecogs.com/gif.latex?c_s) is the category of spam and ![](https://latex.codecogs.com/gif.latex?c_h) is the category of ham. Hence, to decide whether the email is spam or ham, we only need to consider if

> ![decision formula](https://latex.codecogs.com/gif.latex?p%28c_s%29%5Ccdot%20p%28M%20%7C%20c_s%29%20%3E%20p%28c_h%29%5Ccdot%20p%28M%20%7C%20c_h%29)
> ![](https://latex.codecogs.com/gif.latex?%5CLeftrightarrow%20count%28c_h%29%5Ccdot%20p%28M%20%7C%20c_s%29%20%3E%20count%28c_s%29%5Ccdot%20p%28M%20%7C%20c_h%29)

 The formula to calculate ![](https://latex.codecogs.com/gif.latex?p%28M%20%7C%20c_s%29) is:
 > ![](https://latex.codecogs.com/gif.latex?p%28M%20%7C%20c_s%29%20=%20%5Cprod_%7Bi=1%7D%5E%7BnToken%7Dp%28token_i%7Cc_s%29)
 > with ![](https://latex.codecogs.com/gif.latex?p%28token%20%7C%20c_s%29%20=%20%5Cfrac%7B1&plus;count_%7Bspam%7D%28token%29%7D%7B2&plus;count_%7Btotal%7D%28token%29%7D)

The similar goes with ![](https://latex.codecogs.com/gif.latex?p%28M%20%7C%20c_h%29) 

> **Note:**
> In this program, the Long (Bignum type) in python is used to avoid dataloss when using float multiplication. To do that, we multiply each p(token|c) with 1000 and force convert to Long (keep 3 decimal digit).