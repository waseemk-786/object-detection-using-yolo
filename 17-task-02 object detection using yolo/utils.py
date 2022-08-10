
import cv2
import numpy as np

def draw_prediction(frame,classes,classid,confidence,left,top,right,bottom):
    cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0))
    label='%.2f' %confidence
    if classes:
        assert(classid<len(classes))
        label='%s: %s' %(classes[classid],label)
    labelsize,baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_PLAIN,0.5,1)
    top=max(top,labelsize[1])
    cv2.rectangle(
        frame,
        (left,top-labelsize[1]),
        (left+labelsize[0],top+baseline),
        (255,255,255),
        cv2.FILLED
    )
    cv2.putText(
        frame,
        label,
        (left,top),
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        0.5,
        (0,0,0)
    )


def process_frame(frame,outputs,classes,conf,nms):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    classids=[]
    confidences=[]
    boxes=[]
    for out in outputs:
        for detection in out:
            scores=detection[5:]
            classid=np.argmax(scores)
            confidence=scores[classid]
            if confidence>conf:
                center_x=int(detection[0]*frameWidth)
                center_y=int(detection[1]*frameHeight)
                width=int(detection[2]*frameWidth)
                height=int(detection[3]*frameHeight)
                left=int(center_x-(width/2))
                top=int(center_y-(height/2))
                classids.append(classid)
                confidences.append(confidence)
                boxes.append([left,top,width,height])
    indices=cv2.dnn.NMSBoxes(boxes,confidences,conf,nms)
    for i in indices:
        box=boxes[i]
        left=box[0]
        top=box[1]
        width=box[2]
        height=box[3]
        draw_prediction(frame,classes,classids[i],confidences[i],left,top,left+width,top+height)