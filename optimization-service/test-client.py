from __future__ import print_function
import json
import socket
import sys
import struct
# import requests

data = {"batch_sizes": [ 1, 3, 5], "latencies": [44.3, 55.7, 110.4]}

encoded = json.dumps(data).encode('utf-8')
header = struct.pack("<I", len(encoded))
print("SENDING %d bytes" % len(encoded))
message = header + encoded
print(encoded)


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# Connect the socket to the port on the server given by the caller
server_address = ("localhost", 7777)
sock.connect(server_address)
try:
    print(sock.sendall(message))


    header_bytes = 8*2
    data = ""
    # self.request.settimeout(0.5)
    # self.request.setblocking(1)
    # wait for header
    while len(data) < header_bytes:
        data += sock.recv(16)

    alpha, beta = struct.unpack("<dd", data)

    print("ALPHA: %f, BETA: %f" % (alpha, beta))

finally:
    sock.close()
