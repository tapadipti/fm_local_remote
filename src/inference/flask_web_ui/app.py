from flask import Flask, render_template, request
import numpy as np
import boto3, json
from PIL import Image


# This block is needed so that the app can set env vars with AWS credentials
# This is needed when deploying the app in pythonanywhere.com
# Remember to `vim flask_web_ui/.env` in the bash console to save the AWS credentials in the .env file before running this app (in pythonanywhere.com)
# The .env file should look like this:
# AWS_ACCESS_KEY_ID = xxx
# AWS_SECRET_ACCESS_KEY = yyy
# AWS_SESSION_TOKEN = zzz
import os
from dotenv import load_dotenv
project_folder = os.path.expanduser('~/demo-fashion-mnist/src/inference/flask_web_ui')
load_dotenv(os.path.join(project_folder, '.env'))


app = Flask(__name__)
aws_region='us-east-1'
session = boto3.Session()
sagemaker_runtime_client = session.client('runtime.sagemaker', region_name=aws_region)
endpoint_name = 'td-fm-ep-dev'


def pre_process(image_file):
    image = Image.open(image_file)

    # Resize, convert to grayscale, convert to np array and convert to array of required shape
    image = image.resize((28, 28))
    image = image.convert('L')
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image from the form
        image_file = request.files['file']
        
        # Preprocess and convert to list
        image_array = pre_process(image_file)
        image_list = image_array.tolist()

        # Call SageMaker endpoint
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(image_list),
            ContentType='application/json'
        )

        result = response['Body'].read().decode('utf-8')
        result_json = json.loads(result)
        print(result_json)

        prediction = np.array(result_json['predictions'][0])
        predicted_class = np.argmax(prediction)

        # Map the class label to the class name
        label_map = {
            0: "T-shirt or top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
            }
        predicted_class_name = label_map[predicted_class]

        return render_template('index.html', prediction=predicted_class_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
