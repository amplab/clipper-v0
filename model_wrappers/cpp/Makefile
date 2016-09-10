CC=g++ --std=c++0x
noop:
	$(CC) ClipperRPC.cpp NoopModelWrapper.cpp -o noop
vw:
	$(CC) -lvw ClipperRPC.cpp MnistLogisticRegression.cpp -o vw

clean:
	rm vw noop
