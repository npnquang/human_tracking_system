# Introduction
The project consists of **03** versions of Human Tracking system: `DeepSORT`, `DeepOCSORT` and a version that relies only on `Human Re-identification` model.

The project is part of the Internship Program of [COTAI](https://www.cot.ai/) from 19/06/2023 to 12/07/2023.

# Process Description
- Flowchart of the `DeepOCSORT` and `DeepSORT` system:
![](https://hackmd.io/_uploads/HkB4dv5tn.png)

- Flowchart of the `Human Re-identification` system:
![](https://hackmd.io/_uploads/Byt70Yctn.png)


# Installing packages

## DeepOCSORT
The requirements are listed in the `requirement_boxmot.txt` file.
***Note:***
- The `boxmot` module is modified from the original module to fit the project. It is maintained locally so installing boxmot is not required.
- The `ultralytics` module is installed without its dependencies. Hence, it must be installed seperated from the other modules.

1. First, install `ultralytics` without its dependencies.
```
pip install ultralytics==8.0.132 --no-deps
```
2. Next, install the other modules in the `requirement_boxmot.txt` file
```
pip install -r requirement_deepocsort.txt
```

## DeepSORT
The requirements are listed in the `requirement_deepsort.txt` file.

To install the modules in the `requirement_deepsort.txt` file, run the command:
```
pip install -r requirement_deepsort.txt
```

## Human Re-identification
The requirements are listed in the `requirement_reid.txt` file. 

To install the modules in the `requirement_reid.txt` file, run the command:
```
pip install -r requirement_reid.txt
```
# Run the project
After installing the required packages, run the command
```
python deepocsort.py
```
to run the `DeepOCSORT` system.

**OR**

```
python deepsort.py
```
to run the `DeepSORT` system.

**OR**

```
python reid.py
```
to run the `Human Re-identification` system.

# Reference List

- [Human re-identification definition](https://paperswithcode.com/task/person-re-identification#:~:text=or%20image%20sequence.-,It%20involves%20detecting%20and%20tracking%20a%20person%20and%20then%20using,a%20robust%20and%20efficient%20manner.)
- [Pre-trained models ONNX](https://github.com/onnx/models#body_analysis)
- [Pre-trained Reidentification models OpenVino](https://docs.openvino.ai/2023.0/omz_models_group_intel.html#reidentification-models)
- [Pre-trained Object Detection models OpenVINO](https://docs.openvino.ai/2023.0/omz_models_group_intel.html#object-detection-models)
- [Multi Camera Multi Target Python Demo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/multi_camera_multi_target_tracking_demo/python)
- [Simple Online and Realtime Tracking (SORT) paper](https://arxiv.org/pdf/1602.00763.pdf)
- [SORT GitHub](https://github.com/abewley/sort)
- [DeepSORT paper - arxiv.org](https://arxiv.org/abs/1703.07402)
- [DeepSORT GitHub](https://github.com/nwojke/deep_sort)
- [DeepSORT explanation - OpenCV](https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/)

