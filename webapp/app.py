import os
import io
import uuid
import json

from flask import Flask, jsonify, request, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms

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


def get_class_idx_map(metadata_path):
    df = pd.read_csv(metadata_path, index_col='image_id')
    classes = list(df.groupby('dx')['lesion_id'].nunique().keys())
    # cls_idx = {}
    # # for i, cl in enumerate(sorted(classes)):
    #     cls_idx[i] = cl

    # return cls_idx
    return classes


app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.debug = False
wsgiapp = app.wsgi_app

class_name_idx_map = {
    'akiec': "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    'bcc': "basal cell carcinoma",
    'bkl': "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
    'df': "dermatofibroma",
    'mel': "melanoma",
    'nv': "melanocytic nevi",
    'vasc': "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage"
}
class_idx_map = get_class_idx_map(os.path.join(
    'data',
    'HAM10000_metadata.csv'
))
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
    return send_from_directory('templates', 'home.html')


def get_prediction(img_bytes):
    tensor = transform_img(img_bytes=img_bytes)
    output = model.forward(tensor)
    _, y_hat = output.max(1)
    # pred_idx = str(y_hat.item())
    pred_idx = y_hat.item()
    predict = {'lesion_type_index': pred_idx,
               'lesion_type_id': class_idx_map[pred_idx],
               'lesion_type_name': class_name_idx_map[class_idx_map[pred_idx]]}
    return predict


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
