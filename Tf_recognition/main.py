import cv2 as cv
import tf_labels as tf_labels
import sys

tf_labels.initLabels("opencv-extra/ssd_mobilenet_v1_coco.pbtxt")
cvNet = cv.dnn.readNetFromTensorflow("opencv-extra/", "mscoco_label_map.pbtxt")

img = cv.imread(sys.argv[1])
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.25:
        left = int(detection[3] * cols)
        top = int(detection[4] * rows)
        right = int(detection[5] * cols)
        bottom = int(detection[6] * rows)
        label = tf_labels.getLabel(int(detection[1]))
        print(label, score, left, top, right, bottom)
        text_color = (23, 230, 210)
        cv.rectangle(img, (left, top), (right, bottom), text_color, thickness=2)
        cv.putText(img, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

cv.imshow('img', img)
cv.waitKey()