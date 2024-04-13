import cv2
import os
#0 is a primary source
myvideo = cv2.VideoCapture(0)
while True:
    return_value, frame = myvideo.read()
    cv2.imshow("My Video", frame)
    k = cv2.waitKey(10)

    if k == 27:
        break


myvideo.release()