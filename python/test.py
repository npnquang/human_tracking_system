from models import YoLoV8, ReidModel

import openvino.runtime as ov 
from boxmot import DeepOCSORT, StrongSORT
from pathlib import Path
import cv2
import numpy as np
import queue
from threading import Thread
import time

core = ov.Core()
yolo = YoLoV8(core, "..\\models\\yolov8-det\\yolov8n_with_preprocess.xml", convert_color=True)
reid = ReidModel(core, "..\\testmodels\\reid_0287\\person-reidentification-retail-0267.xml")

tracker = StrongSORT(
    embedder=reid,  # which ReID model to use
    device='cpu',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
    fp16=True,  # wether to run the ReID model with half precision or not
    max_age=120,
  
)