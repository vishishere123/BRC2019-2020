"""This python code detects AR tags in a video, and draws green rectangles over the detected tags. To run
the code just input "python AR_Tag_Detector_Rover_Vishal.py --video Demo.mp4"
Here, --video is the flag, and Demo.mp4 is the video name in which there are AR tags"""
import cv2
import numpy
import copy
import imutils
import matplotlib.pyplot as plt
import argparse
def image_process(frame):
    final_contour_list = contour_generator(frame)

    for i in range(len(final_contour_list)):
        cv2.drawContours(frame, [final_contour_list[i]], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", frame)
    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
def contour_generator(frame):
    test_img1 =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_img1,(5,5),0)
    test_canny =cv2.Canny(test_blur, 75, 200)
    edge1 =copy.copy(test_canny)
    cnt,h = cv2.findContours(edge1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index  = []
    contour_list =[]
    for hier in h[0]:
        if hier[3] != -1:
            index.append(hier[3])
    for c in index:
        peri = cv2.arcLength(cnt[c], True)
        approx = cv2.approxPolyDP(cnt[c],0.02*peri,True)

        if len(approx) > 4:
            peri1 = cv2.arcLength(cnt[c - 1], True)
            corners = cv2.approxPolyDP(cnt[c - 1], 0.02 * peri1, True)
            contour_list.append(corners)

    new_contour_list = list()
    for contour in contour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)
    final_contour_list = list()
    for element in new_contour_list:
        if cv2.contourArea(element) < 2500:
            final_contour_list.append(element)

    return final_contour_list
ap = argparse.ArgumentParser()
ap.add_argument("-i","--video",help = "name of video file", required = True)
videos = vars(ap.parse_args())
cap = cv2.VideoCapture(videos["video"])
while cap.isOpened():
    success, frame = cap.read()
    if success == False:
        break
    img =cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
    image_process(img)
cap.release()
