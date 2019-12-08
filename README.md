# Quick start
Visit https://birdid.iscas.ac.cn:8080/upload or http://birdid.iscas.ac.cn:5000/upload.

Send a POST request with a photo to https://birdid.iscas.ac.cn:8080/ or http://birdid.iscas.ac.cn:5000/.

The server will response a json file whose form is: [{"birdNum" :"int","birdNameCN":"str", "probability":"float"}, ...]

# Response examples

```json
{
    "birdExists": true,
    "detected": [
        [
            [
                88.44642639160156,
                108.05941009521484,
                314.1124267578125,
                261.4700622558594
            ],
            [
                {
                    "birdNum": "79",
                    "birdNameCN": "大天鹅",
                    "probability": "72.4"
                },
                {
                    "birdNum": "77",
                    "birdNameCN": "疣鼻天鹅",
                    "probability": "5.4"
                },
                {
                    "birdNum": "179",
                    "birdNameCN": "大白鹭",
                    "probability": "5.2"
                },
                {
                    "birdNum": "293",
                    "birdNameCN": "丹顶鹤",
                    "probability": "3.5"
                },
                {
                    "birdNum": "78",
                    "birdNameCN": "小天鹅",
                    "probability": "2.1"
                }
            ]
        ],
        [
            [
                326.9642333984375,
                112.74176788330078,
                544.1141357421875,
                261.4792785644531
            ],
            [
                {
                    "birdNum": "79",
                    "birdNameCN": "大天鹅",
                    "probability": "75.9"
                },
                {
                    "birdNum": "77",
                    "birdNameCN": "疣鼻天鹅",
                    "probability": "4.2"
                },
                {
                    "birdNum": "157",
                    "birdNameCN": "朱鹮",
                    "probability": "2.9"
                },
                {
                    "birdNum": "293",
                    "birdNameCN": "丹顶鹤",
                    "probability": "2.0"
                },
                {
                    "birdNum": "78",
                    "birdNameCN": "小天鹅",
                    "probability": "1.7"
                }
            ]
        ]
    ]
}
```

### Request (Run `chcp 65001` first if you want to use CMD in order to adjust utf-8 display)

```
curl -F "file=@./1.png" https://birdid.iscas.ac.cn:8080/
curl -F "file=@./1.png" http://birdid.iscas.ac.cn:5000/
```

---

# Code deployment

```
conda env create -f environment.yml
conda activate bird
```

or
```
$ conda env update -f environment.yml
```

# Start & Close

Start
```
conda activate bird

nohup gunicorn -w=8 -t 3600 cas_bird_server:app --certfile ./ssl/birdid.iscas.ac.cn.pem --keyfile ./ssl/birdid.iscas.ac.cn.key -b 0.0.0.0:8080 > ./log/server.log&

nohup gunicorn -b 127.0.0.1:8000 -t 3600 wsgi:app > ./log/server.log&

nohup gunicorn -w=4 -b 127.0.0.1:7000 -t 3600 cas_bird_server_http:app > ./log/server_http.log&
```

Close
```
ps -aux|grep cas_bird_server
pstree -ap|grep gunicorn
kill -9 xxxxxx 
```

---

### BirdSpider

A spider for getting `./resource/target.txt`. Information comes from http://www.birder.cn
