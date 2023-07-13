import openvino.runtime as ov
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import ReidModel, YoLoV8
 
core = ov.Core()
reid_model = ReidModel(core, "..\\models\\intel\\person-reidentification-retail-0287\\FP16-INT8\\person-reidentification-retail-0287.xml")

yolo_model = YoLoV8(core, "..\\models\\yolov8-det\\yolov8n_with_preprocess.xml", convert_color=True)


from deep_sort_realtime.deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort()


# vid = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# vid = cv2.VideoCapture("rtsp://admin:ahihi2022%@192.168.1.162:554/Streaming/Channels/1")
vid = cv2.VideoCapture("..\output_1.avi")
window_name= "display"
cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# vid.set(cv2.CAP_PROP_BUFFERSIZE, 4)
# vid.set(cv2.CAP_PROP_FPS, 20)
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break
    
    # frame = cv2.flip(frame, 1)
    # frame = cv2.resize(frame, (1280, 720))

    # if frame_number == 1: 
    result = yolo_model.forward(frame)
    if len(result) > 0:
        
        if len(result.shape) == 1:
            result = np.expand_dims(result, 0)

        detections = [None] * len(result)
        for i in range(len(result)):
            detection = result[i]
            # cv2.rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (255, 0, 0), 1)
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
            
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            
            cv2.putText(frame, text=str(track_id), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1, \
                thickness=5, org=(int(box[0]), int(box[3])+5))


            
    # Display the resulting frame
    cv2.imshow(window_name, frame)
        # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera from video capture
vid.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows()
