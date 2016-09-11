from __future__ import print_function
import sys, ctypes
from ctypes import c_double, c_uint32, c_uint8, c_char_p, Structure, POINTER
import numpy as np
import time
import datetime
import sys
import os

SHUTDOWN_CODE = 0
FIXEDINT_CODE = 1
FIXEDFLOAT_CODE = 2
FIXEDBYTE_CODE = 3
VARINT_CODE = 4
VARFLOAT_CODE = 5
VARBYTE_CODE = 6
STRING_CODE = 7

class Header(Structure):
    _fields_ = [("code", c_uint8),
                ("num_inputs", c_uint32),
                ("input_len", c_uint32)]

    def __str__(self):
        return "({},{},{})".format(self.code, self.num_inputs, self.input_len)


class ModelWrapperServerS(Structure):
    pass


cur_dir = os.path.dirname(__file__)
dll_path = os.path.join(cur_dir, "faster_rpc_rs/target/release")
prefix = {'win32': ''}.get(sys.platform, 'lib')
extension = {'darwin': '.dylib', 'win32': '.dll'}.get(sys.platform, '.so')
lib_path = os.path.join(dll_path, prefix + "socketserver" + extension)
# print lib_path
# sys.exit(0)
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.init_server.argtypes = (c_char_p, )
lib.init_server.restype = POINTER(ModelWrapperServerS)

lib.server_free.argtypes = (POINTER(ModelWrapperServerS), )

lib.wait_for_connection.argtypes = (POINTER(ModelWrapperServerS), )

lib.get_next_request_header.argtypes = (POINTER(ModelWrapperServerS), )
lib.get_next_request_header.restype = Header

lib.get_fixed_floats_payload.argtypes = (POINTER(ModelWrapperServerS),
                                         POINTER(ctypes.c_double),
                                         c_uint32)

lib.send_response.argtypes = (POINTER(ModelWrapperServerS),
                                         POINTER(ctypes.c_double),
                                         c_uint32)

lib.send_shutdown_message.argtypes = (POINTER(ModelWrapperServerS), )

class ModelWrapperServer:
    def __init__(self, ip, port, model):
        self.obj = lib.init_server("%s:%d" % (ip, port))
        self.model = model

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        lib.server_free(self.obj)

    def serve_forever(self):
        while True:
            lib.wait_for_connection(self.obj)
            self.handle_connection()

    def serve_once(self):
        lib.wait_for_connection(self.obj)
        self.handle_connection()


    def handle_connection(self):
        print("new connection (Python)")
        shutdown = False
        while not shutdown:
            header = lib.get_next_request_header(self.obj)
            if header.code == SHUTDOWN_CODE:
                shutdown = True
                print("Got shutdown message (Python)")
                lib.send_shutdown_message(self.obj)
            else:
                assert header.code == FIXEDFLOAT_CODE
                input_buffer = np.zeros(header.num_inputs * header.input_len)
                assert input_buffer.dtype == np.dtype("float64")
                buffer_pointer = input_buffer.ctypes.data_as(POINTER(c_double))
                lib.get_fixed_floats_payload(self.obj, buffer_pointer, c_uint32(len(input_buffer)))
                input_buffer = input_buffer.reshape(header.num_inputs, header.input_len)
                if np.sum(input_buffer) == 0.0:
                    print("Uh oh, input buffer is still zeroed")
                preds = self.model.predict_floats(input_buffer)
                assert preds.dtype == np.dtype("float64")
                response_buffer_ptr = preds.ctypes.data_as(POINTER(c_double))
                lib.send_response(self.obj, response_buffer_ptr, c_uint32(len(preds)))


def start(model, port):
    ip = "0.0.0.0"
    with ModelWrapperServer(ip, port, model) as server:
        print("Starting to serve (Python)", file=sys.stderr)
        server.serve_forever()
        # server.serve_once()

class ModelWrapperBase(object):
    def predict_ints(self, inputs):
        pass

    def predict_floats(self, inputs):
        pass

    def predict_bytes(self, inputs):
        pass

    def predict_strings(self, inputs):
        pass






