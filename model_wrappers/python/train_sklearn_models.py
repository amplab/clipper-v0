import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.svm as svm
# import sklearn.ensemble.RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
import os
import sys

def load_digits(digits_location, digits_filename = "train.data"):
    digits_path = digits_location + "/" + digits_filename
    print "Source file:", digits_path
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print "Number of image files:", len(data)
    y = data[:,0]
    X = data[:,1:]
    mu = np.mean(X,0)
    sigma = np.var(X,0)
    Z = (X - mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in sigma])
    return (Z, y)


def train_rf(label, num_trees, depths):
    X, y = load_digits("/crankshaw-local/mnist/data")
    my_y = [1. if i == label else 0. for i in y]
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

    # norm_digits_save()
    # digits_loc = "/crankshaw-local/mnist/data"
    #
    # start = int(sys.argv[1])
    # end = int(sys.argv[2])
    # for label in range(start,end + 1):
    #     f_name = "predict_%d_svm" % label
    #     try:
    #         os.mkdir('sklearn_models/%s' % f_name)
    #     except OSError:
    #         print("directory already exists. Might overwrite existing file")
    #     print "training label %d" % label
    #     f = TestFeature(digits_loc, label)
    #     joblib.dump(f, 'sklearn_models/%s/%s.pkl' % (f_name, f_name)) 
    # f = joblib.load('test_model/predict_1_svm.pkl') 
    # print "model trained"
    label = 3
    depths = [1,2,4,8]
    models = train_rf(label, 50, depths)
    test_x, test_y = load_digits("/crankshaw-local/mnist/data", digits_filename="test.data")
    for d in depths:
        pred_wrong = 0.
        pred_total = 500
        for i in range(pred_total):
            idx = np.random.randint(len(test_y))
            y_p = models[d].predict(test_x[idx])[0]
            y_t = test_y[idx] - 1
            if y_t == label:
                y_t = 1.0
            else:
                y_t = 0.0
            if y_t != y_p:
                pred_wrong += 1.
        print "Depth: %d, error: %f" % (d, float(pred_wrong)/float(pred_total))
            




