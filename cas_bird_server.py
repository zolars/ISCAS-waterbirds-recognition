#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

Usage:
    conda env create -f environment.yml
    conda activate bird
    conda install flask pytorch torchvision cudatoolkit=9.0 -c pytorch
    
    https://birdid.iscas.ac.cn:8080/
"""

import os
import time
import sys
from flask import Flask, request, redirect, url_for, render_template
import urllib.parse
import json

import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np

from package.yolov3.detect import detect

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class BCNN(torch.nn.Module):
    """B-CNN for GENDATA.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (172).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 172.
    """
    def __init__(self):
        """Declare all needed layers."""

        torch.nn.Module.__init__(self)

        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(
            *list(self.features.children())[:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 172)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*172.
        """

        N = X.size()[0]
        # assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        # assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        # assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        # assert X.size() == (N, 172)
        return X


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, path):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """

        print('Prepare the network and data...')
        self._path = path

        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()

        # Load the model from disk.
        self._net.load_state_dict(torch.load(self._path['model']))

        self._test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        self.classes = open('./resource/classes.txt',
                            encoding='UTF-8').readlines()

    def test(self, image):

        # Convert gray scale image to RGB image.
        image = image.convert('RGB')

        image = self._test_transforms(image)

        X = torch.autograd.Variable(image.cuda())
        X.unsqueeze_(dim=0)
        # Prediction.
        out = self._net(X)
        out = F.softmax(out, dim=1)
        outnp = out.data[0]

        rstArg = np.argsort(outnp.cpu())
        rstPoss = np.sort(outnp.cpu())

        result = []
        for i in range(5):
            typeStr = self.classes[int(rstArg[-i - 1])]
            result.append({
                "birdNum": typeStr.split()[0],
                "birdNameCN": typeStr.split()[1],
                "probability": '%.1f' % (rstPoss[-i - 1] * 100)
            })

        return result


project_root = os.popen('pwd').read().strip()
path = {
    'model': "./resource/model_172.pth",
    'imgfile': os.path.join(project_root, '1.jpg'),
}
model = BCNNManager(path)

print("\n------------------------Web Start------------------------\n")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def return_img_stream(img_path):
    import base64
    img_stream = ''
    with open(img_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream


# 处理请求
@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename.split(".")[0]

    img_raw = Image.open(file).convert("RGB")
    detected, croppeds = detect(img_raw)

    # No birds in pic
    if "bird" not in detected:
        results = json.dumps({
            "birdExists": False,
            "detected": detected
        },
                             ensure_ascii=False).encode('utf-8')
        print(results)
        return results, 200, {"ContentType": "application/json"}

    # Birds in pic
    results = []
    count = 0
    for coordinate in detected["bird"]:
        img = croppeds[count]
        result = model.test(img)
        results.append([coordinate, result])
        count += 1

    results = json.dumps({
        "birdExists": True,
        "detected": results
    },
                         ensure_ascii=False).encode('utf-8')
    print(results)

    return results, 200, {"ContentType": "application/json"}


# 具有上传功能的页面
@app.route('/upload', methods=['GET', 'POST'])
def upload_test():
    return render_template('upload.html')


@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join("../upload/")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']
    if f:
        fname = f.filename
        ext = fname.rsplit('.', 1)[1]
        unix_time = int(time.time())
        new_filename = str(unix_time) + '.' + ext
        f.save(os.path.join(file_dir, new_filename))  # 保存文件到upload目录

        return json.dumps(({
            "errno": 0,
            "errmsg": "上传成功"
        }), ensure_ascii=False).encode('utf-8')
    else:
        return json.dumps(({
            "errno": 1001,
            "errmsg": "上传失败"
        }),
                          ensure_ascii=False).encode('utf-8')


# @app.route('/upload', methods=['GET', 'POST'])
# @app.route('/upload/', methods=['GET', 'POST'])
# def main_page():
#     if request.method == 'POST':

#         try:
#             shutil.rmtree("./image/")
#             shutil.rmtree("./static/")
#         except OSError as e:
#             print(e)

#         file = request.files['file']
#         if file and allowed_file(file.filename):

#             img = Image.open(file).convert("RGB")
#             os.makedirs("./image/", exist_ok=True)
#             os.makedirs("./static/", exist_ok=True)
#             filename = file.filename.split(".")[0]
#             img.save("./image/" + filename + ".jpg")
#             if "bird" not in detect():
#                 html = '''
#                     <!doctype html>
#                     <title>Error!</title>
#                     <h1>Waterbird doesn't exist in the picture!</h1>
#                     <a href = "/upload"> Continue to upload... </a>
#                 '''
#                 html += "<img src='/static/" + filename + ".png'>"
#                 return html

#             results = model.test(img)

#             html = '''
#                     <!doctype html>
#                     <title>Success!</title>
#                     <h1>Success!</h1>
#                     <a href = "/upload"> Continue to upload... </a>
#                     <p>Your input:</p>
#                     '''
#             html += "<img src='/static/" + filename + ".png'>"

#             html += '''
#                     <p>The result is:</p>
#                     '''
#             count = 0
#             for result in results:
#                 count += 1

#                 birdNameCN = result.get("birdNameCN")
#                 birdNameEN = bd[result.get("birdNum")]['英文名']
#                 birdpicURL = bd[result.get("birdNum")]['pic']

#                 p = result.get("probability")

#                 html += '<p>No.' + str(count) + \
#                     ' : <br>Chinese Name : ' + birdNameCN + '<br>English Name : ' + birdNameEN + \
#                         '<br>Probability : ' + p + '%</p><img src="'+birdpicURL+'" width="300px">'
#             return html

#     return '''
#     <!doctype html>
#     <title>ISCAS Waterbirds Recognition Service</title>
#     <img src="https://raw.githubusercontent.com/zolars/pic-bed/master//20191030165058.png">
#     <h2>Welcome to ISCAS Waterbirds Recognition Service</h2>
#     <h3>Upload new Image</h3>
#     <form action="" method=post enctype=multipart/form-data>
#     <p><input type=file name=file>
#         <input type=submit value=Upload>
#     </form>
#     '''

if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='0.0.0.0',
            port=8080,
            debug=False,
            ssl_context=('./ssl/birdid.iscas.ac.cn.pem',
                         './ssl/birdid.iscas.ac.cn.key'))
