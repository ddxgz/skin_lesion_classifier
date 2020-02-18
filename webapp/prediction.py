import os
import io

import requests
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict

from .densenet import DenseNet

# TODO
MODEL_URL = "https://storage.googleapis.com/data-science-258408-skin-lesion-cls-models/models/dense161.pth.tar"


def load_model(model_dir: str = 'models', force_download: bool = False):
    model_file = os.path.join(model_dir, 'model.pth.tar')

    model_file_existed = os.path.exists(model_file)

    if force_download or not model_file_existed:
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


class LesionPredModel(object):
    def __init__(self, model_dir: str = 'models', force_download: bool = False):
        self.model = load_model(model_dir=model_dir,
                                force_download=force_download)
        self.model.eval()

    def predict(self, img_bytes) -> Dict:
        tensor = transform_img(img_bytes=img_bytes)
        output = self.model.forward(tensor)
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
