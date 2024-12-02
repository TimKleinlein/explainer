import os
import sys
import requests
import numpy as np

from skimage.color import gray2rgb
from keras.datasets import mnist

# Get the list of paths to process from command-line arguments
paths = sys.argv[1:]

# Print the list of paths to process
print('Paths to process:', paths)

# Set the file path for the query data
model_path = paths[0]

# Set the URL for the API endpoint
base = 'http://localhost:5000'
url = base + '/lrp'

# Load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_test[0]
x_rgb = gray2rgb(x)
x_rgb = x_rgb / 255
x = np.expand_dims(x_rgb, axis=0)

# Load model
with open(model_path + '_architecture.json', 'r') as f:
    architecture = f.read()

with open(model_path + '_model.h5', 'rb') as f:
    model_data = f.read()

# Set files for request
files = {'model': model_data}

# Set data for request
data = {'architecture': architecture, 'data': x.tolist()}

# Send the POST request with the image data
response = requests.post(url, data=data, files=files)

# Print the response
print(response.text)
