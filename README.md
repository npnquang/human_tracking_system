# Introduction
The project consists of **03** versions of Human Tracking system: `DeepSORT`, `DeepOCSORT` and a version that relies only on `Human Re-identification` model.


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



