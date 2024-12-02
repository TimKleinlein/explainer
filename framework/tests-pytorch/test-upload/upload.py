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

# Set the URL for the API endpoint
base = 'http://localhost:5000'
url = base + '/saliency'

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

# Load model
model_data = None
with open(model_path, 'rb') as f:
    model_data = ('model', f.read())

query_data.append(model_data)

# Set the data
files = query_data

# Set the sal_params as a dictionary
params = {'abs': False}

# Send the POST request with the image data
response = requests.post(url, files=files, data=params)
# Print the response
print(response.text)
