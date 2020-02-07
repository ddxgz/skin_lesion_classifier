import os
import io
import uuid
import json

from flask import Flask, jsonify, request, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory
import torch
import torchvision.transforms as transforms
from PIL import Image

from .densenet import DenseNet


def load_model():
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = DenseNet(num_classes=7)
    checkpoint = torch.load(os.path.join(
        'models',
        'exp-dense161_eq3_exclutest_lesion_mcc_v1_model_best.pth.tar'),
        map_location=DEVICE
    ) 
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    return net


app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.debug = False
wsgiapp = app.wsgi_app

model = load_model()
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        pred = get_prediction(img_bytes=img_bytes)
        return pred


@app.route('/', methods=['GET'])
def home():
    pass


def get_prediction(img_bytes):
    tensor = transform_img(img_bytes=img_bytes)
    output = model.forward(tensor)
    _, y_hat = output.max(1)
    pred_idx = str(y_hat.item())
    # predict = {}
    return pred_idx


def transform_img(img_bytes):
    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(img_bytes))
    return img_transforms(image).unsqueeze(0)
