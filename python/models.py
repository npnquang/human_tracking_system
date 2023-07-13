import openvino.runtime as ov
import numpy as np
import cv2

# import matplotlib.pyplot as plt
# from ultralytics.yolo.utils import ops
from typing import Tuple, Dict
# import torch
# from PIL import Image

class ReidModel:
    def __init__(self, core, model_path):
        self.core = core
        self.model_path = model_path
        self.model = core.compile_model(model_path, "CPU")
        self.infer_request = self.model.create_infer_request()
        
        # get the input_shape and data_type from the model
        # print(self.model.inputs[0])
        self.input_shape = self.model.inputs[0].partial_shape[2].max_length, self.model.inputs[0].partial_shape[3].max_length # (h, w)
        self.data_type = self.model.inputs[0].element_type.to_dtype()

    
    def __str__(self):
        return f"Model Reidentification:\nInputs: {str(self.model.inputs)} \nOutputs: {str(self.model.outputs)}"

    
    def preprocess(self, frame, bounding_box):
        """
        This function takes in a single frame and multiple bounding box
        This function returns the preprocessed images
        """
        image = frame[int(bounding_box[1]):int(bounding_box[3]), 
                int(bounding_box[0]):int(bounding_box[2])]
        image = np.expand_dims(cv2.resize(image, (self.input_shape[1], self.input_shape[0])).transpose(2, 0, 1),\
            axis=0).astype(self.data_type)
        image = np.ascontiguousarray(image)

        return image


    def forward(self, frame, bounding_boxes):
        """
        This function takes in the frame and the bounding boxes
        This function returns the embedding matrix
        """
        embedding_matrix = np.zeros((len(bounding_boxes), 256), dtype=self.data_type)
        for i in range(len(bounding_boxes)):
            preprocessed_image = self.preprocess(frame, bounding_boxes[i])

            # convert to ov.Tensor
            input_tensor =  ov.Tensor(preprocessed_image, shared_memory=True)

            embedding_matrix[i] = self.infer_request.infer(inputs={"data": input_tensor})["reid_embedding"]

        return embedding_matrix


class YoLoV5Model:
    def __init__(self, core, model_path):
        self.core = core
        self.model_path = model_path
        self.model = core.compile_model(core.read_model(model_path), "CPU")
        self.infer_request = self.model.create_infer_request()

        self.input_shape = self.model.inputs[0].shape[2], self.model.inputs[0].shape[3] # (h, w)
        self.data_type = self.model.inputs[0].element_type.to_dtype()
    
    def __str__(self):
        return f"Model Yolov5:\nInputs: {str(self.model.inputs)} \nOutputs: {str(self.model.outputs)}"

    def resize_letter_box(self, frame, size):
        """
        This function handles the preprocessing process of an image so that it fits the output shape of the yolov5 model
        :param image: numpy array, represents the image
        :param size: int, the desired size to resize the image

        :return image: a numpy array representing the image that has the desired input shape
        """

        # takek the shape of the image out
        h, w, _ = frame.shape
        
        new_image = np.full((size, size, 3), 114, dtype=np.uint8)
        padding = np.zeros((2,)) # left padding and up padding
        
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
            padding[0] = pad_first
            

        elif h < w:
            diff = size - int(size/w*h)
            
            pad_first = diff//2
            pad_second = diff - pad_first

            frame = cv2.resize(frame, (size, int(size/w*h)), interpolation=cv2.INTER_LINEAR)
            new_image[pad_first-1:size-pad_second-1, :, :] = frame

            padding[1] = pad_second

        return new_image, padding

    
    def preprocess(self, frame, size):
        # resize
        frame, padding = self.resize_letter_box(frame, size)
        # convert color
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
        # normalize
        frame = frame / 255.0

        # convert color channel and transpose
        frame = np.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(self.data_type)
        frame = np.ascontiguousarray(frame)
        return frame, padding

    
    def calculate_iou(self, boxes1, boxes2):

        # check if the boxes variables has correct shape
        if len(boxes1.shape) == 1:
            boxes1 = np.expand_dims(boxes1, axis=0)
        if len(boxes2.shape) == 1:
            boxes2 = np.expand_dims(boxes2, axis=0)

        boxes1, boxes2 = boxes1.astype("int"), boxes2.astype("int")

        # Calculate the intersection coordinates
        inter_x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

        # Calculate the areas of intersection and union
        intersection_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        boxes1_area = (boxes1[:, 2]- boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2]- boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = boxes1_area + boxes2_area - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area
        return iou


    def nms(self, result):
        # sort the result in the decreasing order of the bounding box conf
        conf = result[:, 4]
        sorted_ids = np.argsort(conf*-1)
        result = result[sorted_ids]

        selected = [] # a list containing the selected bounding box

        while len(result) > 0:
            # take out the selected box and append it to the selected list
            selected_box = result[0, :]

            selected.append(selected_box)

            # take out the other boxes 
            boxes = result[1:, :4]
            result = result[1:, :]

            # calculate the iou between the selected_box and the other boxes
            iou = self.calculate_iou(selected_box[:4], boxes)
            
            # filter the bounding boxes with small iou
            result = result[iou <= 0.45]

        return np.array(selected) if len(selected) > 0 else None
    

    def scale_box(self, boxes, padding, h, w):

        # check if the boxes variable has the correct shape
        if len(boxes.shape) == 1:
            boxes = np.expand_dims(boxes, axis=0)

        # subtract the offset and then compute the value of the bounding boxes according to the original size of the image
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - padding[0]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - padding[1]
        
        # limit the values of bounding boxes between 0 and 640
        boxes[:, :4] = np.clip(boxes[:, :4], 0, 640)

        boxes[:, :4] = boxes[:, :4] / 640 * max(h, w)

        
        return boxes
    

    def postprocess(self, result, padding, h, w):
        # remove the boxes with low confidence and filter out the person class
        result = result[(result[:, 4] >= 0.8)]

        # take out the confidence of boxes and classes
        box_confs = result[:, 4:5]
        class_confs = result[:, 5:]

        # find the class of each b_box
        classes = np.expand_dims(np.argmax(box_confs, axis=-1), -1)

        # find the conf value of the class for each b_box
        class_conf_value = np.expand_dims(np.max(class_confs, axis=-1), -1)
        # compute the net conf = box_confs * class_conf_value
        conf_value = class_conf_value * box_confs
        # concatenate the results
        # the new_result now has the following format: x_center, y_center, width, height, class, conf
        result = np.concatenate((result[:, :4], conf_value, classes), axis=-1)

        # filter the boxes with low net confidence
        result = result[(result[:, 5] == 0) & (result[:, 4] >= 0.6)]

        # check the length of result after filtering
        if len(result) == 0:
            return None

        # change the format of the bounding boxes from xywh to xyxy
        # the result now has the format x1, y1, x2, y2, conf, class
        result[:, 0] = result[:, 0] - result[:, 2]/2 
        result[:, 1] = result[:, 1] - result[:, 3]/2

        result[:, 2] = result[:, 2] + result[:, 0]
        result[:, 3] = result[:, 3] + result[:, 1]  

        # check the length of result after filtering
        if result.shape[0] > 0:
            # apply the NMS algorithm to the result
            result = self.nms(result)

            # if the returned result is a numpy array, rescale these bounding box to the original size of the image
            if type(result) == np.ndarray:
                result = self.scale_box(result, padding, h, w)

        elif result.shape[0] == 0:
            return None
        return result if len(result) <= 10 else result[:10]
    
    def forward(self, frame):
        h, w, _ = frame.shape

        frame, padding = self.preprocess(frame, self.input_shape[0])

        input_tensor = ov.Tensor(frame, shared_memory=True)
        result = self.infer_request.infer(inputs={"images": input_tensor})["output0"][0]
        result = self.postprocess(result, padding, h, w)
        
        return result

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
        self.infer_request = self.model.create_infer_request()
        self.convert_color = convert_color

        self.min_conf_threshold = min_conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.agnosting_nms = agnosting_nms
        self.max_detections = max_detections
        self.pred_masks = pred_masks
        self.retina_mask = retina_mask

        self.input_shape = self.model.inputs[0].shape[2], self.model.inputs[0].shape[3] # (h, w)
        self.data_type = self.model.inputs[0].element_type.to_dtype()
    
    def forward(self, image):
        """
        OpenVINO YOLOv8 model with integrated preprocessing inference function. Preprocess image, runs model inference and postprocess results using NMS.
        Parameters:
            image (np.ndarray): input image.
            model (Model): OpenVINO compiled model.
        Returns:
            detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
        """
        if self.convert_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = self.resize_letter_box(image, self.input_shape[0])
        input_hw = image_resize.shape[:2]
        input_tensor = np.expand_dims(image_resize, 0)

        result = self.infer_request.infer(inputs={"images": input_tensor})["output0"]
        detections = self.postprocess(result, input_hw, image)
        return detections

    def postprocess(self,
        pred_boxes:np.ndarray,
        input_hw:Tuple[int, int],
        orig_img:np.ndarray):
        """
        YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
        Parameters:
            pred_boxes (np.ndarray): model output prediction boxes
            input_hw (np.ndarray): preprocessed image
            orig_image (np.ndarray): image before preprocessing
            min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
            nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
            max_detections (int, *optional*, 300):  maximum detections after NMS
            pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
            retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
        Returns:
        pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
        """
        nms_kwargs = {"agnostic": self.agnosting_nms, "max_det":self.max_detections}
        # if pred_masks is not None:
        #     nms_kwargs["nm"] = 32
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
        """
        This function handles the preprocessing process of an image so that it fits the output shape of the yolov5 model
        :param image: numpy array, represents the image
        :param size: int, the desired size to resize the image

        :return image: a numpy array representing the image that has the desired input shape
        """

        # takek the shape of the image out
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


if __name__ == "__main__":
    core = ov.Core()
    reid_model = ReidModel(core, "..\models\intel\person-reidentification-retail-0287\FP16-INT8\person-reidentification-retail-0287.xml")

    yolo_model = YoLoV8(core, "..\\models\\yolov8-det\\yolov8n_with_preprocess.xml", convert_color=True)

    image = cv2.imread(filename="D:\\Internship_CoTAI\\src\\5981885-a-group-of-young-people-walking-down-a-street-in-a-large-city.jpg")



    result = yolo_model.forward(image)
    print(result)
    for i in result:
        cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 1)

    cv2.imshow("something", image)

    cv2.waitKey(0)
    
    # It is for removing/deleting created GUI window from screen
    # and memory
    cv2.destroyAllWindows()
    embedding_matrix = reid_model.forward(image, result)
    print(embedding_matrix.shape)
    print(embedding_matrix)



    
    