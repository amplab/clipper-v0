from __future__ import print_function
import numpy as np
import time
import datetime
import sys
import os

import rpc


class NoopModelWrapper(rpc.ModelWrapperBase):


    def __init__(self):
        print("NoopModelWrapper init code running")


    def predict_ints(self, inputs):
        print("predicting %d integer inputs" % len(inputs))
        return np.arange(1,len(inputs) + 1).astype('float64')

    def predict_floats(self, inputs):
        print("predicting %d float inputs" % len(inputs))
        return np.arange(1,len(inputs) + 1).astype('float64')


if __name__=='__main__':
    model = NoopModelWrapper()
    rpc.start(model, "0.0.0.0", 6001)
