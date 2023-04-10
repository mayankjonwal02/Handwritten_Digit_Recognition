import os
from turtle import mode
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mediapipe as mp
model = tf.keras.models.load_model("handwritten.model")
class art():
    def __init__(self):
        self.hand_detector=mp.solutions.hands.Hands()
        self.drawing_utils=mp.solutions.drawing_utils
        self.points=[]
        self.cap = cv2.VideoCapture(0)
        self.drawart()
    def drawart(self):
        while True:
            _,frame=self.cap.read()
            frame=cv2.flip(frame,1)

            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            output=self.hand_detector.process(rgb_frame)
            hands=output.multi_hand_landmarks
            fheight,fwidth,_=frame.shape
            if hands:
                for hand in hands:
                    #drawing_utils.draw_landmarks(frame,hand)
                    landmarks=hand.landmark
                    for id,landmark in enumerate(landmarks):
                        x=int(landmark.x*fwidth)
                        y=int(landmark.y*fheight)

                        if id==4:
                            thumby=y
                            thumbx=x
                        if id ==8:
                            indexy=y
                            indexx=x
                            if abs(thumby-indexy)<30:
                                cv2.circle(frame,(indexx,indexy),30,(0,0,255))
                                self.points.append((indexx,indexy))
                                print(x,y)
                        
                        if id==20:
                            littlex=x
                            littley=y
                        if id==12:
                            middley=y
                            middlex=x
                            if abs(thumbx-middlex)<20:
                                color=(255,0,0)

                        if id==16:
                            ringy=y
                            ringx=x
                            if abs(thumbx-ringx)<20:
                                color=(0,0,255)
            if cv2.waitKey(1) == ord('r'):
                            
                            #if abs(indexy-littley)<10 and littlex>150 and littlex < 500 and littley > 100:
                            #cv2.circle(frame,(littlex,littley),30,(0,0,255))
                            self.points=[]
            if cv2.waitKey(1) == ord('q'):
                break

            
            h,w,c = frame.shape
            black_image = np.zeros((h,w,c), dtype=np.uint8)
            for i in self.points:
                cv2.circle(black_image,i,10,(255,255,255),-1)
                cv2.circle(frame,i,10,(255,255,255),-1)
            # Resize the image using cv2.resize()
            resized_image = cv2.resize(black_image, (28, 28))
            # Reshape the resized image to match the input shape of the Keras model
            resized_image = np.expand_dims(resized_image[:,:,0], axis=-1)
            resized_image = np.expand_dims(resized_image, axis=0)
            
            prediction = model.predict(resized_image)
            print(np.argmax(prediction))

            cv2.imshow('virtual marker',frame)
            cv2.waitKey(1)

                        

                        


            


a=art()

