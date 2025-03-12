# importing libraries
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Hand Gestures Number Counting in Left Hand
cap=cv2.VideoCapture(0)
hand=mp.solutions.hands
Hand=hand.Hands(max_num_hands=1)
mpdraw=mp.solutions.drawing_utils
while True:
    check,img=cap.read()
    results=Hand.process(img)
    handspoints=results.multi_hand_landmarks
    h,w,_=img.shape
    pointer=[]
    if handspoints:
        for points in handspoints:
            mpdraw.draw_landmarks(img,points,hand.HAND_CONNECTIONS)
            for id,cord in enumerate(points.landmark):
                cx,cy=int(cord.x*w),int(cord.y*h)
                cv2.rectangle(img,(80,10),(180,100),(255,255,255),-1)
                cv2.putText(img,str(id),(cx,cy+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,250),1)
                pointer.append((cx,cy))
        dist=[8,12,16,20]
        count=0
        if points:
            if pointer[4][0]<pointer[2][0]:
                count+=1
            for x in dist:
                if pointer[x][1]< pointer[x-2][1]:
                    count+=1
        cv2.putText(img,str(count),(100,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,250,0),2)           
    cv2.imshow("L I V E",img)
    if cv2.waitKey(1)&0xff==ord("a"):
        break
cap.release()
cv2.destroyAllWindows()