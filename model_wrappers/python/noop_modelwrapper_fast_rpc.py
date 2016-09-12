from __future__ import print_function
import numpy as np
import time
import datetime
import sys
import os
import faster_rpc
import os
import sys



class NoopModelWrapper(faster_rpc.ModelWrapperBase):

    def __init__(self):
        pass

    def predict_ints(self, inputs):
        return np.ones(len(inputs))

    def predict_floats(self, inputs):
        # print("batch: %d" % len(inputs))
        return np.ones(len(inputs))


if __name__=='__main__':
    # model_path = os.environ["CLIPPER_MODEL_PATH"]
    # print(model_path, file=sys.stderr)
    model = NoopModelWrapper()
    faster_rpc.start(model, 6001)
