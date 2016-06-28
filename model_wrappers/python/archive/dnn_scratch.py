
import graphlab as gl 
import graphlab.numpy
import pandas as pd
import os




def load_digits(digits_location, digits_filename = "train-mnist-dense-with-labels.data"):
    digits_path = digits_location + "/" + digits_filename
    print "Source file:", digits_path
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print "Number of image files:", len(data)
    y = data[:,0]
    X = data[:,1:]
    return (X, y)


def main():
    model_path = 'dato_model/neuralnet.model'
    model = gl.load_model(model_path)
    mnist_path = os.path.expanduser("~/model-serving/data/mnist_data")
    X, y = load_digits(mnist_path, "test-mnist-dense-with-labels.data")

    first_x = X[1]
    data = gl.SFrame(first_x)
    data['image'] = graphlab.image_analysis.resize(data['image'], 256, 256, 3)
    fs = model.extract_features(xxx)

    print fs




if __name__=='__main__':
    main()
