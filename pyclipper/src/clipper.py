from __future__ import print_function
import sys, ctypes
from ctypes import c_double, c_uint32, c_char_p, Structure, POINTER
import os
import sys
import numpy as np
import time

cur_dir = os.path.dirname(__file__)
dll_path = os.path.join(cur_dir, "../target/release")

class PyClipperS(Structure):
    pass

prefix = {'win32': ''}.get(sys.platform, 'lib')
extension = {'darwin': '.dylib', 'win32': '.dll'}.get(sys.platform, '.so')
lib_path = os.path.join(dll_path, prefix + "pyclipper" + extension)
# print lib_path
# sys.exit(0)
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.init_clipper.argtypes = (c_char_p, )
lib.init_clipper.restype = POINTER(PyClipperS)

lib.pyclipper_free.argtypes = (POINTER(PyClipperS), )

lib.pyclipper_predict.argtypes = (POINTER(PyClipperS), POINTER(ctypes.c_double), c_uint32)
lib.pyclipper_predict.restype = c_double


class PyClipper:
    def __init__(self, config):
        self.obj = lib.init_clipper(config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        lib.pyclipper_free(self.obj)

    def predict(self, x):
        # x must be 1 1d numpy array
        assert len(x.shape) == 1
        x_p = x.ctypes.data_as(POINTER(c_double))
        return lib.pyclipper_predict(self.obj, x_p, c_uint32(len(x)))

with PyClipper("../features.toml") as clipper:
    print(clipper.predict(np.ones(7)*2.2))
    time.sleep(10)
    print(clipper.predict(np.ones(7)*-1.3))
    time.sleep(10)
