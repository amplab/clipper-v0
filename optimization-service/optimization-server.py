from __future__ import print_function
import array
import struct
import SocketServer
import numpy as np
import time
import datetime
import sys
import os
import json
import pandas as pd
import statsmodels.formula.api as smf


class QuantileRegressionHandler(SocketServer.BaseRequestHandler):

    allow_reuse_address = True

    def handle(self):
        print("Handling")
        header_bytes = 4
        data = ""
        while len(data) < header_bytes:
            data += self.request.recv(4096)

        header, data = (data[:header_bytes], data[header_bytes:])
        num_bytes = struct.unpack("<I", header)[0]

        while len(data) < num_bytes:
            data += self.request.recv(4096)

        parsed_data = json.loads(data)
        alpha, beta = fit_quantile_regression(parsed_data["batch_sizes"], parsed_data["latencies"])
        encoded = json.dumps({"alpha": alpha, "beta": beta}).encode('utf-8')
        print(encoded)
        header = struct.pack("<I", len(encoded))
        message = header + encoded
        self.request.sendall(message)


def fit_quantile_regression(batches, latencies):
    quantile = 0.99
    df = pd.DataFrame({"batch_size": batches, "latencies": latencies})
    mod = smf.quantreg('latencies ~ batch_size', df)
    model_fit = mod.fit(quantile)
    alpha = model_fit.params['batch_size']
    beta = model_fit.params['Intercept']
    # NOTE: y = alpha * x + beta
    return (alpha, beta)




def start(port):
    ip = "0.0.0.0"
    server = SocketServer.TCPServer((ip, port), QuantileRegressionHandler)
    # server.handle_request()
    print("Starting to listen for optimizations", file=sys.stderr)
    server.serve_forever()


if __name__=='__main__':
    start(7003)
