# Preprocess the image and convert it to the format expected by the model
# For example, resizing the image to (28, 28) and converting it to a NumPy array

import numpy as np
from PIL import Image

# Load the image
image_file = 'inference_samples/tshirt.png'  # Replace with the path to your image file
image = Image.open(image_file)

# Resize the image to (28, 28)
image = image.resize((28, 28))

# Convert the image to grayscale (if it's not already)
image = image.convert('L')

# Convert the PIL image to a NumPy array
image_array = np.array(image)

# Normalize the pixel values to [0, 1] as the model expects inputs in this range
image_array = image_array / 255.0

# Reshape the image to match the input shape of the model (num_samples, height, width, channels)
image_array = image_array.reshape(1, 28, 28, 1)


# Make predictions using the deployed endpoint
import json
import boto3
session = boto3.Session()
sagemaker_runtime = session.client("runtime.sagemaker")

image_bytes = json.dumps(image_array.tolist())

endpoint_name = 'td-fm-ep-dev' # Replace this with the deployment endpoint name for the required stage.
response = sagemaker_runtime.invoke_endpoint(EndpointName=endpoint_name, Body=image_bytes, ContentType='application/json')

if response['ResponseMetadata']['HTTPStatusCode'] == 200:
    inference_result = response['Body'].read().decode('utf-8')
    print('Inference result:', inference_result)
else:
    print('Error:', response['ResponseMetadata']['HTTPStatusCode'], response['Body'].read().decode('utf-8'))