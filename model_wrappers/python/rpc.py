from __future__ import print_function
import array
import struct
import SocketServer
import numpy as np
import time
import datetime
import sys
import os
import lz4

SHUTDOWN_CODE = 0
FIXEDINT_CODE = 1
FIXEDFLOAT_CODE = 2
FIXEDBYTE_CODE = 3
VARINT_CODE = 4
VARFLOAT_CODE = 5
VARBYTE_CODE = 6
STRING_CODE = 7


# class NoopModelWrapper(ModelWrapperBase):
#
#
#     def __init__(self):
#         print("NoopModelWrapper init code running")
#
#
#     def predict_ints(self, inputs):
#         print("predicting %d integer inputs" % len(inputs))
#         return np.arange(1,len(inputs) + 1)
#
#     def predict_floats(self, inputs):
#         print("predicting %d float inputs" % len(inputs))
#         return np.arange(1,len(inputs) + 1)

class ModelWrapperBase(object):
    def predict_ints(self, inputs):
        pass

    def predict_floats(self, inputs):
        pass

    def predict_bytes(self, inputs):
        pass

    def predict_strings(self, inputs):
        pass

def is_fixed_format(fmt):
    return fmt == FIXEDINT_CODE or fmt == FIXEDFLOAT_CODE or fmt == FIXEDBYTE_CODE

def is_var_format(fmt):
    return fmt == VARINT_CODE or fmt == VARFLOAT_CODE or fmt == VARBYTE_CODE

class ClipperRpc(SocketServer.BaseRequestHandler):

    allow_reuse_address = True

    # def __init__(self):
        # self.allow_reuse_address = True



    def handle(self):
        print("HANDLING NEW CONNECTION")
        while True:
            header_bytes = 5
            data = ""
            # self.request.settimeout(0.5)
            # self.request.setblocking(1)
            # wait for header
            while len(data) < header_bytes:
                data += self.request.recv(4096)

            header, data = (data[:header_bytes], data[header_bytes:])
            input_type, num_inputs = struct.unpack("<BI", header)
            if input_type == SHUTDOWN_CODE:
                print("Shutting down connection")
                self.request.sendall(np.array([1234]).astype('uint32').tobytes())
                return
            if is_fixed_format(input_type):
                additional_header_bytes = 4
                while len(data) < additional_header_bytes:
                    data += self.request.recv(4096)
                input_len = struct.unpack("<I", data[:additional_header_bytes])[0]
                data = data[additional_header_bytes:]
                inputs = []
                if input_type == FIXEDBYTE_CODE:
                    total_bytes_expected = input_len*num_inputs
                    while len(data) < total_bytes_expected:
                        data += self.request.recv(4096)
                    input_bytes = np.array(array.array('B', bytes(data)))
                    inputs = np.split(input_bytes, num_inputs)
                    for i in inputs:
                        assert len(i) == input_len
                elif input_type == FIXEDFLOAT_CODE:
                    total_bytes_expected = 8*input_len*num_inputs
                    while len(data) < total_bytes_expected:
                        data += self.request.recv(4096)
                    input_doubles = np.array(array.array('d', bytes(data)))
                    inputs = np.split(input_doubles, num_inputs)
                    for i in inputs:
                        assert len(i) == input_len
                else:
                    assert input_type == FIXEDINT_CODE
                    total_bytes_expected = 4*input_len*num_inputs
                    while len(data) < total_bytes_expected:
                        data += self.request.recv(4096)
                    input_ints = np.array(array.array('i', bytes(data)))
                    inputs = np.split(input_ints, num_inputs)
                    for i in inputs:
                        assert len(i) == input_len

            elif is_var_format(input_type):
                additional_header_bytes = 4
                while len(data) < additional_header_bytes:
                    data += self.request.recv(4096)
                content_len = struct.unpack("<I", data[:additional_header_bytes])[0]
                data = data[additional_header_bytes:]
                inputs = []
                while len(data) < content_len:
                    data += self.request.recv(4096)
                for i in range(0, num_inputs):
                    input_len = struct.unpack("<I", data[:4])[0]
                    data = data[4:] 
                    if input_type == VARBYTE_CODE:
                        inputs.append(data[:input_len])
                    else:
                        # Our inputs are either four byte floats or integers, so we must read
                        # 4 * input_len bytes from the packed data
                        input_len = input_len * 4
                        if input_type == VARFLOAT_CODE:
                            inputs.append(np.array(array.array('f', bytes(data[:input_len]))))
                        else:
                            assert input_type == VARINT_CODE
                            inputs.append(np.array(array.array('i', bytes(data[:input_len]))))
                    data = data[input_len:]
                assert len(inputs) == num_inputs

            elif input_type == STRING_CODE:
                additional_header_bytes = 4
                while len(data) < additional_header_bytes:
                    data += self.request.recv(4096)
                content_len = struct.unpack("<I", data[:additional_header_bytes])[0]
                data = data[additional_header_bytes:]
                while len(data) < content_len:
                    data += self.request.recv(4096)
                inputs = []
                input_lengths_bytes = 4 * num_inputs
                input_lengths = np.array(array.array('i', bytes(data[:input_lengths_bytes])))
                data = data[input_lengths_bytes:]
                decompressed_strs = lz4.loads(data)
                for length in input_lengths:
                    inputs.append(decompressed_strs[:length])
                    decompressed_strs = decompressed_strs[length:]
                for i in range(0, len(inputs)):
                    assert len(inputs[i]) == input_lengths[i]
            else:
                raise RuntimeError("Invalid input type: " + input)

            if input_type == FIXEDINT_CODE or input_type == VARINT_CODE:
                predictions = self.server.model.predict_ints(inputs)
            elif input_type == FIXEDFLOAT_CODE or input_type == VARFLOAT_CODE:
                predictions = self.server.model.predict_floats(inputs)
            elif input_type == FIXEDBYTE_CODE or input_type == VARBYTE_CODE:
                predictions = self.server.model.predict_bytes(inputs)
            else:
                predictions = self.server.model.predict_strings(inputs)
            assert len(predictions) == num_inputs
            assert predictions.dtype == np.dtype('float64')
            self.request.sendall(predictions.tobytes())




def start(model_wrapper, ip, port):
    server = SocketServer.TCPServer((ip, port), ClipperRpc)
    server.model = model_wrapper
    # server.handle_request()
    print("Starting to serve")
    server.serve_forever()



