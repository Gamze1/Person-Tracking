# Person Tracking-YOLOv3-Deep Sort-yTorch

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
#Test Video
Test video is given in the link:https://drive.google.com/file/d/1EQWVibPefvQWv6aeJZyn4l16n_1zP0g4/view?usp=sharing
