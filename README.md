# An Assistive Learning Workflow on Annotating Images for Object Detection
## Paper link: http://eil.stanford.edu/publications/vivian_wong/assistive_learning.pdf

YOLOv3 in this implementation is based on https://github.com/ultralytics/yolov3

Mask RCNN in this implementation is based on https://github.com/maxkferg/metal-defect-detection

## Instructions: 
1. Create conda environment: 
```
conda env create -f environment.yml
```
2. Start Flask
```
export FLASK_APP=app.py
export FLASK_ENV='development'
export FLASK_DEBUG=0 
```
Note that DEBUG mode must be off. A tensorflow-keras error will occur if it is on (https://github.com/tensorflow/tensorflow/issues/34607)

