from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn import svm
# import sklearn.ensemble.RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
import os
import sys

def load_digits(digits_location, digits_filename = "train.data", norm=True):
    digits_path = digits_location + "/" + digits_filename
    print("Source file: %s" % digits_path)
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print("Number of image files: %d" % len(data))
    y = data[:,0]
    X = data[:,1:]
    Z = X
    if norm:
        mu = np.mean(X,0)
        sigma = np.var(X,0)
        Z = (X - mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in sigma])
    return (Z, y)


def train_rf(label, num_trees, depths):
    X, y = load_digits("/crankshaw-local/mnist/data")
    my_y = [1. if i == label else -1.0 for i in y]
    models = {}
    for d in depths:
        model = RFC(n_estimators=num_trees, max_depth=d)
        model.fit(X, my_y)
        fname = "%drf_pred%d_depth%d" % (num_trees, label, d)
        try:
            os.mkdir('sklearn_models/%s' % fname)
        except OSError:
            print("directory already exists. Might overwrite existing file")
        joblib.dump(model, 'sklearn_models/%s/%s.pkl' % (fname, fname)) 
        models[d] = model
    return models

def train_svm(label):
    fname = "svm_pred%d" % (label)
    path = 'sklearn_models/%s/%s.pkl' % (fname, fname)
    print("Training svm model")
    X, y = load_digits("/crankshaw-local/mnist/data")
    my_y = [1. if i == label else -1.0 for i in y]
    model = svm.SVC()
    model.fit(X, my_y)
    try:
        os.mkdir('sklearn_models/%s' % fname)
    except OSError:
        print("directory already exists. Might overwrite existing file")
    joblib.dump(model, path) 
    print("Done training svm model")
    # model = joblib.load(path)
    return model

def train_logistic_regression(label):
    fname = "log_regression_pred%d" % (label)
    path = 'sklearn_models/%s/%s.pkl' % (fname, fname)
    print("Training logistic regression")
    X, y = load_digits("/crankshaw-local/mnist/data")
    my_y = [1. if i == label else -1.0 for i in y]
    model = lm.LogisticRegression()
    model.fit(X, my_y)
    try:
        os.mkdir('sklearn_models/%s' % fname)
    except OSError:
        print("directory already exists. Might overwrite existing file")
    joblib.dump(model, path) 
    print("Done training logistic regression model")
    # model = joblib.load(path)
    return model


# # Predicts if digit is 1
# class TestFeature:
#
#     def __init__(self, digits_loc, label):
#         self.label = label + 1
#         X, y = load_digits(digits_loc)
#         self.mu = np.mean(X,0)
#         self.sigma = np.var(X,0)
#         Z = self.normalize_digits(X)
#         my_y = [1. if i == self.label else 0. for i in y]
#         print np.count_nonzero(my_y)
#
#         model = svm.SVC()
#         model.fit(Z, my_y)
#         self.model = model
#
#     def normalize_digits(self, X):
#         Z = (X - self.mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in self.sigma])
#         return Z
#         
#     def predict(self, x):
#         z = self.normalize_digits(x)
#         return self.model.predict(z)
#     
#     def hash_input(self, x):
#         return hash(x.data.tobytes())


# def norm_digits_save():
#     digits_loc = "/crankshaw-local/mnist/data"
#     digits_filename = "train.data"
#     out_filename = "train_norm.data"
#     digits_path = digits_loc + "/" + digits_filename
#     print "Source file:", digits_path
#     df = pd.read_csv(digits_path, sep=",", header=None)
#     data = df.values
#     print "Number of image files:", len(data)
#     y = data[:,0]
#     X = data[:,1:]
#     mu = np.mean(X,0)
#     sigma = np.var(X,0)
#     Z = (X - mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in sigma])
#     with open(digits_loc + "/" + out_filename, "w") as of:
#         for i in range(len(y)):
#             z_str = [str(j) for j in Z[i].tolist()]
#             line = str(y[i]) + "," + ",".join(z_str) + "\n"
#             of.write(line)



if __name__=='__main__':

    label = 3
    depths = [2, 4, 8, 16]
    train_rf(label, 50, depths)
    models = {"lr": train_logistic_regression(label), "svm": train_svm(label)}
    # test_x, test_y = load_digits("/crankshaw-local/mnist/data", digits_filename="test.data", norm=True)
    # for m in models:
    #     pred_wrong = 0
    #     num_pos_labels = 0
    #     num_neg_labels = 0
    #     pos_predicted = 0
    #     neg_predicted = 0
    #     pred_total = len(test_y)
    #     for idx in range(len(test_y)):
    #         # idx = np.random.randint(len(test_y))
    #         y_p = models[m].predict(test_x[idx].reshape(1, -1))[0]
    #         y_t = test_y[idx] - 1
    #         if y_t == label:
    #             y_t = 1.0
    #             num_pos_labels += 1
    #         else:
    #             y_t = -1.0
    #             num_neg_labels += 1
    #         if y_t != y_p:
    #             pred_wrong += 1
    #         if y_p == -1.0:
    #             neg_predicted += 1
    #         else:
    #             pos_predicted += 1
    #
    #     print("Model: %s, error: %f, pos_predicted: %d, neg_predicted: %d, pos_labels: %d, neg_labels: %d" % (m,
    #             float(pred_wrong)/float(pred_total),
    #             pos_predicted,
    #             neg_predicted,
    #             num_pos_labels,
    #             num_neg_labels))
    #         
    #
    #
    #
    #
