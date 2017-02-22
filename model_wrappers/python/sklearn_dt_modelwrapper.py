
from __future__ import print_function
import numpy as np
import time
import datetime
import sys
import os
# import rpc
import faster_rpc
import pandas as pd
from sklearn import linear_model as lm
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
import os
import sys



class SklearnModelWrapper(faster_rpc.ModelWrapperBase):


    def __init__(self, path):
        success = False
        self.model = joblib.load(path) 
        # print("Loaded %s model" % type(self.model))
        print("Loaded %s model" % type(self.model), file=sys.stderr)
        self.path = path


    def predict_ints(self, inputs):
        preds = self.model.predict(inputs)
        print("sum: %f, len: %d" % (preds.sum(), len(preds)), file=sys.stderr)
        return np.array(preds == 0).astype(np.float64)

    def predict_floats(self, inputs):
        preds = self.model.predict(inputs)
        print("sum: %f, len: %d" % (preds.sum(), len(preds)), file=sys.stderr)
        return np.array(preds == 0).astype(np.float64)




if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    pkl_names = [l for l in os.listdir(model_path) if os.path.splitext(l)[1] == ".pkl"]
    assert len(pkl_names) == 1
    pkl_path = os.path.join(model_path, pkl_names[0])
    print(pkl_path, file=sys.stderr)
    model = SklearnModelWrapper(pkl_path)
    faster_rpc.start(model, 6001)
