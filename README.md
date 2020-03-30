# Quick start

Send a POST request with a photo to https://birdid.iscas.ac.cn:8080/ and http://birdid.iscas.ac.cn:5000/.

The server will response a json file whose form such as

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
gunicorn -c gunicorn.conf.py cas_bird_server:app
```

Close
```
pkill gunicorn
```

---

### BirdSpider

A spider for getting `./resource/target.txt`. Information comes from http://www.birder.cn
