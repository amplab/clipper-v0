import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

## Getting the data
#
# brew install s3cmd
# ./s3cmd --configure
# ./s3cmd get s3://mnist-data/train-mnist-dense-with-labels.data .
# ./s3cmd get s3://mnist-data/test-mnist-dense-with-labels.data .
#

class Task:
    def __init__(self, pref, X, y, test_X, test_y, digit_class_labels):
        self.pref = pref
        self.X = X
        self.y = y
        self.test_X = test_X
        self.test_y = test_y
        self.digit_class_labels = digit_class_labels
        self.cluster = None
    def __str__(self):
        return "pref: %d, y: %s, X: %s, test_y: %s, test_X: %s" % (self.pref,
                                                                   str(self.y),
                                                                   str(self.X),
                                                                   str(self.test_y),
                                                                   str(self.test_X))
    
    def __repr__(self):
        return str(self)

    

def load_digits(digits_location, digits_filename = "train-mnist-dense-with-labels.data"):
    digits_path = digits_location + "/" + digits_filename
    print "Source file:", digits_path
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print "Number of image files:", len(data)
    y = data[:,0]
    X = data[:,1:]
    return (X, y)

def to_image(x):
    return np.reshape(x,[28,28])

def display_digit(x):
    plt.imshow(to_image(x), interpolation='none')

def display_random_digits(X, y):
    ind = np.random.permutation(len(X))
    plt.figure()
    for i in range(0, 16):
        plt.subplot(4,4,i+1)
        display_digit(X[ind[i],:])
        plt.draw()
        # Display the plot
  

def normalize_digits(X):
    mu = np.mean(X,0)
    sigma = np.var(X,0)
    Z = (X - mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in sigma])
    return Z 

def fourier_project(X, nfeatures = 4096, scale = 1.e-4):
    (n,d) = X.shape
    W = np.random.normal(scale = scale, size = [d, nfeatures])
    phase = np.random.uniform( size = [1, nfeatures]) * 2.0 * np.pi
    randomFeatures = np.cos(X.dot(W) + phase)
    return randomFeatures

def filter_two_class(X, y, digitA = 3, digitB = 9):
    yInd = (y == (digitA + 1)) | (y == (digitB + 1))
    yBinary = (y == (digitA + 1)) * 1.
    return (yInd, yBinary[yInd])


def train_test_split(y, propTrain = 0.75):
    ind = np.random.permutation(len(y))
    split_ind = ind[0.75 * len(y)]
    train_ind = ind[:split_ind]
    test_ind = ind[split_ind:]
    print "Train size: ", len(train_ind)
    print "Train true: ", np.mean(y[train_ind] == 1.0)
    print "Test size:  ", len(test_ind)
    print "Test true:  ", np.mean(y[test_ind] == 1.0)
    return (train_ind, test_ind)


class OracleModel:

    def __init__(self, train_xs, train_ys, test_xs, test_ys, pref_digit):
        self.train_xs = train_xs
        self.train_ys = train_ys
        self.test_xs = test_xs
        self.test_ys = test_ys
        self.pref_digit = pref_digit
        self.model = None

def gen_oracle_xs(pref_digit, num_examples):
    split = int(num_examples / 2.)
    true_xs = [np.zeros(10) for j in range(0, split)]
    for x in true_xs:
        x[pref_digit] = 1
    false_xs = [np.zeros(10) for j in range(0, split)]
    for x in false_xs:
        non_pref = np.random.randint(0,10)
        while non_pref == pref_digit:
            non_pref = np.random.randint(0,10)
        x[non_pref] = 1
    all_x = np.concatenate((true_xs, false_xs), axis = 0)
    all_y = np.concatenate((np.ones(split), np.zeros(split)), axis=0)
    shuffle_perm = np.random.permutation(len(all_x))
    xs = all_x[shuffle_perm, :]
    ys = all_y[shuffle_perm]
    return (xs, ys)


def create_oracle_datasets(nTasks=100, taskSize=100, testSize=100):
    tasks = []
    split = taskSize / 2
    for i in range(0, nTasks):
        if i % 50 == 0:
            print "making task", i
        prefDigit = np.random.randint(0,10)
        (train_xs, train_ys) = gen_oracle_xs(prefDigit, taskSize)
        (test_xs, test_ys) = gen_oracle_xs(prefDigit, testSize)
        tasks.append(OracleModel(train_xs, train_ys, test_xs, test_ys, prefDigit))
    return tasks


def create_mtl_datasets(X, y, nTasks=1000, taskSize=100, testSize=100): 
    tasks = []
    for i in range(0, nTasks):
        if i % 50 == 0:
            print "making task", i
        prefDigit = np.random.randint(0,10)
        
        tSplit = int(taskSize * 0.5)
        fSplit = taskSize - tSplit

        test_tSplit = int(testSize * 0.5)
        test_fSplit = testSize - test_tSplit
        perm = np.random.permutation(len(y)) # shuffle dataset
        inClass = y == (prefDigit + 1) # bitmask of examples in preferred class
        t_ind = np.flatnonzero(inClass[perm]) # positive examples
        f_ind = np.flatnonzero(~inClass[perm]) # negative examples
        tX = X[perm[t_ind[:tSplit]], :] # take first taskSize/2 true examples from shuffled dataset
        fX = X[perm[f_ind[:fSplit]], :] # take first taskSize/2 false examples from shuffled dataset
        tX_digit_class = y[perm[t_ind[:tSplit]]]
        fX_digit_class = y[perm[f_ind[:tSplit]]]
        digit_class_labels = np.concatenate((tX_digit_class, fX_digit_class), axis=0)
        newX = np.concatenate((tX, fX), axis = 0)
        newy = np.concatenate((np.ones(tSplit), np.zeros(fSplit)), axis = 0)
        shuffle_perm = np.random.permutation(len(newy)) # shuffle selected training examples
        newX = newX[shuffle_perm, :]
        newy = newy[shuffle_perm]
        digit_class_labels = digit_class_labels[shuffle_perm]
        test_tX = X[perm[t_ind[tSplit:(tSplit + test_tSplit)]], :]
        test_fX = X[perm[f_ind[fSplit:(fSplit + test_fSplit)]], :]
        test_newX = np.concatenate((test_tX, test_fX), axis = 0)
        test_newy = np.concatenate((np.ones(test_tSplit), np.zeros(test_fSplit)), axis = 0)
        test_shuffle_perm = np.random.permutation(len(test_newy))
        test_newX = test_newX[test_shuffle_perm, :]
        test_newy = test_newy[test_shuffle_perm]
        task = Task(prefDigit, newX, newy, test_newX, test_newy, digit_class_labels)
        tasks += [task]
    return tasks



