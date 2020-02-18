import os
import io
import uuid
import json

from flask import Flask, jsonify, request, send_from_directory, abort
from PIL import Image
from typing import List, Dict

from .prediction import LesionPredModel

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.debug = True
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024
SUPPORTED_IMAGE_TYPE: List[str] = ['JPEG', "JPG"]
wsgiapp = app.wsgi_app

# avoid down model when in debug
# from_url: bool = True
# if app.debug:
#     from_url = False

# model = prediction.load_model(force_download=from_url)
# model.eval()
model = LesionPredModel()

# avoid forcely set to True when import this module
# app.debug = False


def supported_image(filename: str) -> bool:
    if '.' not in filename:
        return False

    suffix = filename.rsplit('.', 1)[1]

    if suffix.upper() not in SUPPORTED_IMAGE_TYPE:
        return False

    return True


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files.get('file')
        if not img_file:
            abort(400)
        if not supported_image(img_file.filename):
            abort(415)
        img_bytes = img_file.read()
        pred: Dict = model.predict(img_bytes=img_bytes)
        return pred


@app.route('/', methods=['GET'])
def home():
    return send_from_directory('vuejs', 'home.html')
