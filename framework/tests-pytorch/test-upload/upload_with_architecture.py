import os
import sys
import requests
import torch
import json


# Get the list of paths to process from command-line arguments
paths = sys.argv[1:]

# Print the list of paths to process
print('Paths to process:', paths)

# Set the file path for the query data
model_path = paths[0]

# Set the file path for the query data
query_path = paths[1]

# Set the file path for the query data
architecture_path = paths[2]

# Set the URL for the API endpoint
base = 'http://localhost:5000'
url = base + '/occlusion'

# Check if path is a folder
if os.path.isdir(query_path):
    # If path is a folder, load all files into array
    files = []
    for file in os.listdir(query_path):
        files.append(os.path.join(query_path, file))
else:
    # If path is not a folder, load file into array
    files = [query_path]

# Load files into an array
query_data = []
for file in files:
    with open(file, 'rb') as f:
        query_data.append(('data', f.read()))

# Load model dic
model_data = None
with open(model_path, 'rb') as f:
    model_data = ('model', f.read())

query_data.append(model_data)

# load architecture
architecture_data = None
with open(architecture_path, 'rb') as f:
    architecture_data = ('architecture', f.read())

query_data.append(architecture_data)

# Set the data
files = query_data

# example for sending a tensor as parameter
rand_tensor = torch.randn(2, 3, 32, 32, requires_grad=True)
serialized_tensor = json.dumps(rand_tensor.tolist())

# Set the params as a dictionary
params = {'sliding_window_shapes': (3, 10, 10),
          'baselines': serialized_tensor}

# Send the POST request with the image data
response = requests.post(url, files=files, data=params)

# Print the response
print(response.text)
