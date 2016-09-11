from __future__ import print_function
import numpy as np
import time
import datetime
import sys
import os
import rpc
import os
import sys



class NoopModelWrapper(rpc.ModelWrapperBase):


    def __init__(self):
        pass


    def predict_ints(self, inputs):
        return np.ones(len(inputs))
        # preds = self.model.predict(inputs)
        # print("sum: %f, len: %d" % (preds.sum(), len(preds)), file=sys.stderr)
        # return preds
        # return self.model.predict(inputs)
        # for x in inputs:
        #     preds.append(float(self.model.predict(x)))
        # # print("predicting %d integer inputs" % len(inputs))
        # return np.array(preds)

    def predict_floats(self, inputs):
        return np.ones(len(inputs))
        # preds = self.model.predict(inputs)
        # print("sum: %f, len: %d" % (preds.sum(), len(preds)), file=sys.stderr)
        # # print("sum: %f, len: %d" % (preds.sum(), len(preds)))
        # return preds
        # preds = []
        # for x in inputs:
        #     preds.append(float(self.model.predict(x)))
        # return np.array(preds)




if __name__=='__main__':
    # model_path = os.environ["CLIPPER_MODEL_PATH"]
    # print(model_path, file=sys.stderr)
    model = NoopModelWrapper()
    rpc.start(model, 6001)
