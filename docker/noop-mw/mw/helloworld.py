from __future__ import print_function
import array
import struct
import SocketServer
import numpy as np
import time
import datetime
import sys
import os


class HWServer(SocketServer.BaseRequestHandler):

    allow_reuse_address = True

    # def __init__(self):
        # self.allow_reuse_address = True



    def handle(self):
        print("HANDLING NEW CONNECTION")
        # while True:
        header_bytes = 5
        data = ""
        # self.request.settimeout(0.5)
        # self.request.setblocking(1)
        # wait for header
        while len(data) < header_bytes:
            data += self.request.recv(4096)

        print("got some data: %s" % data)
        self.request.sendall("ljjsdhfksdf")
        return




def start(ip, port):
    server = SocketServer.TCPServer((ip, port), HWServer)
    # server.handle_request()
    print("HELLO WORLD SERVER STARTING")
    server.serve_forever()

if __name__=='__main__':
    start("0.0.0.0", 6001)


