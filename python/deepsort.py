import openvino.runtime as ov
import numpy as np
import cv2

from models import ReidModel, YoLoV5Model
from deep_sort_realtime.deepsort_tracker import DeepSort
 
core = ov.Core()
reid_model = ReidModel(core, "..\\models\\intel\\person-reidentification-retail-0287\\FP16-INT8\\person-reidentification-retail-0287.xml")

yolo_model = YoLoV5Model(core, "..\\models\\yolo5n_openvino_model\\yolov5n.xml")


tracker = DeepSort(max_age=5)

vid = cv2.VideoCapture(0)
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # if frame_number == 1: 
    result = yolo_model.forward(frame)
    if type(result) == np.ndarray:
        detections = [None] * len(result)
        for i in range(len(result)):
            detection = result[i]
            left = detection[0]
            top = detection[1]
            w = detection[2] - detection[0]
            h = detection[3] - detection[1]
            detections[i] = ([left, top, w, h], detection[4], detection[5])

        embedding = reid_model.forward(frame, result)

        tracks = tracker.update_tracks(detections, embeds=embedding)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            box = track.to_ltrb()

            cv2.rectangle(frame, (int(box[2]), int(box[1])), (int(box[0]), int(box[3])), (255, 0, 0), 1)
            cv2.putText(frame, text=str(track_id), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1, \
                thickness=5, org=(int(box[0]), int(box[1])))


            
    # Display the resulting frame
    cv2.imshow('Tracking System', frame)
        # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera from video capture
vid.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows()
