import sys
import os
from package.yolov3.detect_bird import detect

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
detect(image_folder="data")
