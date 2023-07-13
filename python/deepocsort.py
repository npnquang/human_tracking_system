import openvino.runtime as ov
import numpy as np
import cv2

from ultralytics.yolo.utils import ops
from typing import Tuple, Dict
import torch
from PIL import Image

from models import ReidModel
from boxmot import DeepOCSORT
import queue
from threading import Thread
import time


class YoLoV8Model:
    def __init__(self, core, model_path, convert_color=True, 
        min_conf_threshold:float = 0.6,
        nms_iou_threshold:float = 0.7,
        agnosting_nms:bool = False,
        max_detections:int = 300,
        pred_masks:np.ndarray = None,
        retina_mask:bool = False):
        self.core = core
        self.model_path = model_path
        self.model = core.compile_model(model_path, "CPU")

        self.current_infer_request = self.model.create_infer_request()
        self.next_infer_request = self.model.create_infer_request()

        self.convert_color = convert_color

        self.min_conf_threshold = min_conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.agnosting_nms = agnosting_nms
        self.max_detections = max_detections
        self.pred_masks = pred_masks
        self.retina_mask = retina_mask

        self.input_shape = self.model.inputs[0].shape[2], self.model.inputs[0].shape[3] # (h, w)
        self.data_type = self.model.inputs[0].element_type.to_dtype()
    

    def preprocess(self, image):
        if self.convert_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = self.resize_letter_box(image, self.input_shape[0])
        input_hw = image_resize.shape[:2]
        input_tensor = np.expand_dims(image_resize, 0)
        return input_hw, input_tensor

    def postprocess(self,
        pred_boxes:np.ndarray,
        input_hw:Tuple[int, int],
        orig_img:np.ndarray):
        
        nms_kwargs = {"agnostic": self.agnosting_nms, "max_det":self.max_detections}

        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            self.min_conf_threshold,
            self.nms_iou_threshold,
            classes=[0],
            nc=80,
            **nms_kwargs
        )
        results = []
        proto = torch.from_numpy(self.pred_masks) if self.pred_masks is not None else None

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append([])
                continue
            if proto is None:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                results.append(pred.numpy())
                continue
            if self.retina_mask:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
                segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
            results.append(pred[:, :6].numpy())
        return np.array(results).squeeze()

    def resize_letter_box(self, frame, size):
        h, w, _ = frame.shape
        
        new_image = np.full((size, size, 3), 114, dtype=np.uint8)
        
        if h > w:
            # calculate the difference
            diff = size - int(size/h*w)

            # calculate the first and second pad
            pad_first = diff//2
            pad_second = diff - pad_first

            # resize the image
            frame = cv2.resize(frame, (int(size/h*w), size), interpolation=cv2.INTER_LINEAR)
            # padding
            new_image[:, pad_first-1:size-pad_second-1, :] = frame
            

        elif h < w:
            diff = size - int(size/w*h)
            
            pad_first = diff//2
            pad_second = diff - pad_first

            frame = cv2.resize(frame, (size, int(size/w*h)), interpolation=cv2.INTER_LINEAR)
            new_image[pad_first-1:size-pad_second-1, :, :] = frame


        return new_image

class CameraThreading:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.capture = capture
        self.max_queue_length = max_queue_length

        self.frames_queue = queue.Queue(maxsize=self.max_queue_length)

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() >= self.max_queue_length:
                self.frames_queue.get()
            ret, frame = self.capture.read()

            if not ret and self.frames_queue.empty():
                self.process = False
                break
            if ret:
                self.frames_queue.put(frame, block=False)


core = ov.Core()
yolo = YoLoV8Model(core, "..\\models\\yolov8-det\\yolov8n_with_preprocess.xml", convert_color=True)
reid = ReidModel(core, "..\\models\\dynamic_input\\reid_0287\\person-reidentification-retail-0267.xml")

# roi = np.array([[550, 200], [800, 200],
#             [1000, 500], [800, 500]],
#            np.int32).reshape(-1, 1, 2)

tracker = DeepOCSORT(
    embedder=reid,  # which ReID model to use
    device='cpu',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
    fp16=True,  # wether to run the ReID model with half precision or not
    max_age=100 ,
    roi=None) 


color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5
window_name= "Tracking System"

model_input = yolo.model.inputs[0]

vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture("rtsp://admin:ahihi2022%@192.168.1.162:554/Streaming/Channels/1")
# vid = cv2.VideoCapture("..\\MOT16-04-raw.webm")
# cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
#                         cv2.WINDOW_FULLSCREEN)
thread_body = CameraThreading(vid, max_queue_length=30)
frames_thread = Thread(target=thread_body)
frames_thread.start()

frame = thread_body.frames_queue.get()

# frame = cv2.flip(frame, 1)
input_hw, input_tensor = yolo.preprocess(frame)

yolo.current_infer_request.set_tensor(model_input, ov.Tensor(input_tensor.astype(np.uint8)))
yolo.current_infer_request.start_async()


while(True):
    start = time.time()
    # if thread_body.frames_queue.empty():
    #     break
    next_frame = thread_body.frames_queue.get()

    # next_frame = cv2.flip(next_frame, 1)
    next_input_hw, next_input_tensor = yolo.preprocess(next_frame)
    yolo.next_infer_request.set_tensor(model_input, ov.Tensor(next_input_tensor.astype(np.uint8)))

    yolo.next_infer_request.start_async()

    if yolo.current_infer_request.wait_for(-1) == 1:
        results = yolo.current_infer_request.get_output_tensor(0).data
        detections = yolo.postprocess(results, input_hw, frame)

        if type(tracker.roi) == np.ndarray:
            frame =  cv2.polylines(frame, [tracker.roi], isClosed=True, color = (255, 0, 0), thickness = 2)
        if len(detections) > 0:
            if len(detections.shape) == 1:
                detections = np.expand_dims(detections, 0)

            ts = tracker.update(detections, frame)
            # print(ts)
            if ts.shape[0] != 0:
                xyxys = ts[:, 0:4].astype('int') # float64 to int
                ids = ts[:, 4].astype('int') # float64 to int
                confs = ts[:, 5]
                clss = ts[:, 6]
                frame_times = ts[:, 7]
                centers = ts[:, 8:10]
                # print bboxes with their associated id, cls and conf
            
                for xyxy, id, conf, clss_num, frame_time, center in zip(xyxys, ids, confs, clss, frame_times, centers):
                    frame = cv2.circle(frame, (int(center[0]), int(center[1])), radius=5, color=(0, 0, 255), thickness=thickness)
                    cv2.putText(
                        frame,
                        f'id: {id} time: {int(frame_time//23)} second',
                        (xyxy[0], xyxy[3]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        color,
                        thickness
                    )


    # Display the resulting frame
    cv2.imshow(window_name, frame)

    frame = next_frame
    input_hw = next_input_hw

    yolo.current_infer_request, yolo.next_infer_request = yolo.next_infer_request, yolo.current_infer_request

    stop = time.time()
    fps = round(1/(stop-start), 2)
    # print(f"FPS: {fps}")

        # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
thread_body.process = False
frames_thread.join()
thread_body.capture.release()
cv2.destroyAllWindows()



