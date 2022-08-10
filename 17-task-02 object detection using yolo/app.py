import cv2
import numpy as np
from utils import process_frame,draw_prediction
confidence=0.3
nms=0.4
with open ('coco.names','rt') as f:
    classes=f.read().rstrip('\n').split('\n')

net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg","darknet")
layernames=net.getUnconnectedOutLayersNames()
cam=cv2.VideoCapture(0)

while True:
    status,frame=cam.read()
    if status:
        blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
        net.setInput(blob)
        outs=net.forward(layernames)
        process_frame(frame,outs,classes,confidence,nms)
        cv2.imshow('result',frame)
        cv2.waitKey(1)
                