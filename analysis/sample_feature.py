import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.svm as svm
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
    return (X, y)


# Predicts if digit is 1
class TestFeature:

    def __init__(self, digits_loc, label):
        self.label = label + 1
        X, y = load_digits(digits_loc)
        self.mu = np.mean(X,0)
        self.sigma = np.var(X,0)
        Z = self.normalize_digits(X)
        my_y = [1. if i == self.label else 0. for i in y]
        print np.count_nonzero(my_y)

        model = svm.SVC()
        model.fit(Z, my_y)
        self.model = model

    def normalize_digits(self, X):
        Z = (X - self.mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in self.sigma])
        return Z
        
    def predict(self, x):
        z = self.normalize_digits(x)
        return self.model.predict(z)
    
    def hash_input(self, x):
        return hash(x.data.tobytes())

if __name__=='__main__':
    digits_loc = "/crankshaw-local/mnist/data"
    
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    for label in range(start,end + 1):
        f_name = "predict_%d_svm" % label
        try:
            os.mkdir('sklearn_models/%s' % f_name)
        except OSError:
            print("directory already exists. Might overwrite existing file")
        print "training label %d" % label
        f = TestFeature(digits_loc, label)
        joblib.dump(f, 'sklearn_models/%s/%s.pkl' % (f_name, f_name)) 
    # f = joblib.load('test_model/predict_1_svm.pkl') 
    # print "model trained"
    # test_x, test_y = load_digits(digits_loc, digits_filename="test-mnist-dense-with-labels.data")
    # pred_wrong = 0.
    # pred_total = 200
    # for i in range(pred_total):
    #     idx = np.random.randint(len(test_y))
    #     y_p = f.predict(test_x[idx])[0]
    #     y_t = test_y[idx] - 1
    #     if y_t == f.label:
    #         y_t = 1.0
    #     else:
    #         y_t = 0.0
    #     if y_t != y_p:
    #         pred_wrong += 1.
    # print float(pred_wrong)/float(pred_total)
            




