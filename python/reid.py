import openvino.runtime as ov
import numpy as np
import cv2

from models import ReidModel, YoLoV5Model
from track import Track, ClusterFeature
from tracker import SingleCameraTracker

core = ov.Core()
reid_model = ReidModel(core, "..\\models\\intel\\person-reidentification-retail-0287\\FP16-INT8\\person-reidentification-retail-0287.xml")

yolo_model = YoLoV5Model(core, "..\\models\\yolo5n_openvino_model\\yolov5n.xml")



tracker = SingleCameraTracker(reid_model, MAX_NUM_TRACK=30, NUM_CLUSTERS=4, TRACKING_THRESHOLD=0.7, MIN_TRACK_LENGTH=3)

# define a video capture object
vid = cv2.VideoCapture(0)

frame_number = 0

while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    # print(frame.shape)

    # if frame_number == 1: 
    result = yolo_model.forward(frame)
    if type(result) == np.ndarray:
    

        updated_index = tracker.process(frame, result)
        # print(updated_index)
        for j in range(len(result)):
            box = result[j]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            cv2.putText(frame, text=str(updated_index[j]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1, \
                thickness=5, org=(int(box[0]), int(box[1])))

            # frame_number = 0
    # else:
    #     frame_number += 1 

    # Display the resulting frame
    cv2.imshow('Tracking System', frame)
        # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break