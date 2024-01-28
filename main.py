from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import cv2
import requests
import numpy as np
from io import BytesIO

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class peopleCount(Resource):
    def get(self):
        img=cv2.imread('images/zdj.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {'count': len(boxes)}
class pclink(Resource):
    def get(self):
       image_url = request.args.get('image_url')
       response = requests.get(image_url)
       img = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
       boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
       return {'count': len(boxes)}

class pcprzeslanie(Resource):
    def post(self):
        file = request.files['file']
        image_stream = BytesIO(file.read())
        img = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), -1)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {'count': len(boxes)}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(peopleCount, '/dysk')
api.add_resource(HelloWorld, '/test')
api.add_resource(pclink, '/link')
api.add_resource(pcprzeslanie, '/przeslanie')

if __name__ == '__main__':
    app.run(debug=True)