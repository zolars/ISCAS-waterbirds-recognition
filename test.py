import sys
import os
import shutil
from PIL import Image
from package.yolov3.detect import detect

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


img = Image.open("./1.png").convert("RGB")
os.makedirs("./image/", exist_ok=True)
img.save("./image/1.jpg")

print(detect())

img = Image.open("./static/1.png")

try:
    shutil.rmtree("./image/")
    shutil.rmtree("./static/1/")
    os.remove("./static/1.png")
except OSError as e:
    print(e)
