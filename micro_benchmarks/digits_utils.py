import pandas as pd
import numpy as np

def load_digits(digits_location, digits_filename = "train-mnist-dense-with-labels.data"):
    digits_path = digits_location + "/" + digits_filename
    print "Source file:", digits_path
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print "Number of image files:", len(data)
    y = data[:,0]
    X = data[:,1:]
    return (X, y)

def normalize_digits(X):
    mu = np.mean(X,0)
    sigma = np.var(X,0)
    Z = (X - mu) / np.array([np.sqrt(z) if z > 0 else 1. for z in sigma])
    return Z 

def fourier_project(X, nfeatures = 4096, scale = 1.e-4):
    (n,d) = X.shape
    W = np.random.normal(scale = scale, size = [d, nfeatures])
    phase = np.random.uniform( size = [1, nfeatures]) * 2.0 * np.pi
    randomFeatures = np.cos(X.dot(W) + phase)
    return randomFeatures

