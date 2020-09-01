# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:40:23 2020

@author: Bibek77
"""


from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import  numpy as np

face_classifier=cv2.CascadeClassifier(r'D:\FacialExpressionRecognition\haarcascade_frontalface_alt.xml')
model= load_model(r'D:\FacialExpressionRecognition\Emotion.h5')

class_labels=['Angry', 'Happy','sad', 'netural', 'suprised']

Capture=cv2.VideoCapture(0)


while True:
    # capture the frame
    ret, frame = Capture.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
#draw the rectangle iin face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        R_gray = gray[y:y+h,x:x+w]
        R_gray = cv2.resize(R_gray,(48,48),interpolation=cv2.INTER_AREA)
    


        if np.sum([R_gray])!=0:
            RegionToDetect = R_gray.astype('float')/255.0
            RegionToDetect = img_to_array(RegionToDetect)
            RegionToDetect = np.expand_dims(RegionToDetect,axis=0)

   

            preds = model.predict(RegionToDetect)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Facial Expression Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Capture.release()
cv2.destroyAllWindows()
    

    