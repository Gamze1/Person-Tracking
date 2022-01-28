# Person Tracking-YOLOv3-Deep Sort-PyTorch

## Introduction

This repository consist of PyTorch,YOLOv3 and Deepsort. 
Person tracking is carried out with the Deepsort algorithm which is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector.

## Requirements

- python
- torch >= 1.3
- torch-vision
- opencv-python
- pillow


## Tracking

track.py runs tracking on any video source:

```bash
python track.py --source ...
```
