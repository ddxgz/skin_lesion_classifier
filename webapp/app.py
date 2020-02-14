import os
import io
import uuid
import json

import requests
from flask import Flask, jsonify, request, send_from_directory, abort
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from typing import List, Dict

from .densenet import DenseNet

MODEL_URL = "https://storage.googleapis.com/data-science-258408-skin-lesion-cls-models/models/dense161.pth.tar"


def load_model(model_dir: str = 'models', from_url: bool = True):
    model_file = os.path.join(model_dir, 'model.pth.tar')

    if from_url:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(model_file, 'wb') as f:
            resp = requests.get(MODEL_URL, allow_redirects=True)
            f.write(resp.content)

    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = DenseNet(num_classes=7)
    checkpoint = torch.load(
        model_file,
        map_location=DEVICE
    )
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    return net


def get_class_idx_map(metadata_path: str) -> List[str]:
    df = pd.read_csv(metadata_path, index_col='image_id')
    classes: List[str] = list(df.groupby('dx')['lesion_id'].nunique().keys())
    # cls_idx = {}
    # # for i, cl in enumerate(sorted(classes)):
    #     cls_idx[i] = cl

    # return cls_idx
    return classes


app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.debug = False
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024
SUPPORTED_IMAGE_TYPE: List[str] = ['JPEG', "JPG"]
wsgiapp = app.wsgi_app

class_name_idx_map: Dict[str, str] = {
    'akiec': "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    'bcc': "basal cell carcinoma",
    'bkl': "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
    'df': "dermatofibroma",
    'mel': "melanoma",
    'nv': "melanocytic nevi",
    'vasc': "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage"
}
class_idx_map: List[str] = get_class_idx_map(os.path.join(
    'data',
    'HAM10000_metadata.csv'
))
model = load_model()
model.eval()


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
        pred: Dict = get_prediction(img_bytes=img_bytes)
        return pred


@app.route('/', methods=['GET'])
def home():
    return send_from_directory('vuejs', 'home.html')


def get_prediction(img_bytes) -> Dict:
    tensor = transform_img(img_bytes=img_bytes)
    output = model.forward(tensor)
    _, y_hat = output.max(1)
    # pred_idx = str(y_hat.item())
    pred_idx: int = y_hat.item()
    predict = {'lesion_type_index': pred_idx,
               'lesion_type_id': class_idx_map[pred_idx],
               'lesion_type_name': class_name_idx_map[class_idx_map[pred_idx]]}
    return predict


def transform_img(img_bytes) -> torch.Tensor:
    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(img_bytes))
    return img_transforms(image).unsqueeze(0)
