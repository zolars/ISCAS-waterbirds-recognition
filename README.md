### 测试命令 (如果使用 CMD 要先执行 `chcp 65001` 调整utf-8显示)

`curl -F "file=@./1.png" https://birdid.iscas.ac.cn:8080/`
`curl -F "file=@./1.png" http://birdid.iscas.ac.cn:5000/`

### 使用方式

向 https://birdid.iscas.ac.cn:8080/ 或者 http://birdid.iscas.ac.cn:5000/ 发送一个包含图片文件的 POST 指令.

服务器返回一个 json 文件, 格式为：[{"birdNum" :"int","birdNameCN":"str", "probability":"float"}, ...]

### 网页示例
https://birdid.iscas.ac.cn:8080/upload
http://birdid.iscas.ac.cn:5000/upload

### 返回示例

```json
[
    {"birdNum": "77", "birdNameCN": "疣鼻天鹅", "probability": "98.1"}, 
    {"birdNum": "79", "birdNameCN": "大天鹅", "probability": "0.9"}, 
    {"birdNum": "78", "birdNameCN": "小天鹅", "probability": "0.2"}, 
    {"birdNum": "187", "birdNameCN": "斑嘴鹈鹕", "probability": "0.1"}, 
    {"birdNum": "157", "birdNameCN": "朱鹮", "probability": "0.1"}
]
```
---

### Code deployment

```
conda env create -f environment.yml
conda activate bird
```

or
```
$ conda env update -f environment.yml
```

### Start & Close

Start
```
conda activate bird

nohup gunicorn -b 127.0.0.1:8000 -t 3600 src.cas_bird_server:app > ./log/server.log&

nohup gunicorn -b 127.0.0.1:7000 -t 3600 src.cas_bird_server_http:app > ./log/server_http.log&
```

Close
```
ps -aux|grep cas_bird_server
kill -9 xxxxxx 
```

---

### BirdSpider

一个可以抓取 `./resource/target.txt` 中拉丁文名的鸟种的信息. 信息来源自 http://www.birder.cn

---

### Bilinear-CNN Training

Mean field approximation of Bilinear CNN for Fine-grained recognition
* Step 1. Fine-tune the fc layer only.
    It gives 75.47% accuracy on CUB.
    ```
    $ CUDA_VISIBLE_DEVICES=0 python ./src/get_conv.py
    $ nohup python ./src/train.py --base_lr 1e0 --batch_size 64 --epochs 80 --weight_decay 1e-5 > ./log/train_fc.log&
    ```

* Step 2. Fine-tune all layers.
    It gives 84.41% accuracy on CUB.
    ```
    $ nohup python ./src/train.py --base_lr 1e-2 --batch_size 64 --epochs 80 --weight_decay 1e-5 --pretrained "bcnn_fc_epoch_best.pth" > ./log/train_all.log&
    ```

