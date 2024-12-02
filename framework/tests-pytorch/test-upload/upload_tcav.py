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
# NOTE: concepts zip file has to be named concepts.zip!
concept_path = paths[0]

# Set the file path for the query data
# NOTE: input zip file has to be named input_images.zip!
input_path = paths[1]

# Set the file path for the query data
model_path = paths[2]

# Set the URL for the API endpoint
base = 'http://localhost:5000'
url = base + '/tcav'

query_data = []

# Load model
model_data = None
with open(model_path, 'rb') as f:
    model_data = ('model', f.read())

query_data.append(model_data)

# load architecture
#architecture_data = None
#with open(architecture_path, 'rb') as f:
#    architecture_data = ('architecture', f.read())

#query_data.append(architecture_data)

# load concepts
zip_data = None
with open(concept_path, 'rb') as f:
    zip_data = ('concepts', f.read())

query_data.append(zip_data)

# load input images
zip_data = None
with open(input_path, 'rb') as f:
    zip_data = ('input_images', f.read())

query_data.append(zip_data)


# Set the params as a dictionary
params = {'layers': ['inception4c', 'inception4d', 'inception4e']}


# send request
response = requests.post(url, files=query_data, data=params)
print(response.text)
