import cv2
import mediapipe as mp
import csv
from collections import deque

CSV_FILE="/home/abdelali/Downloads/dataset.csv"

cap=cv2.VideoCapture(0)
hands=mp.solutions.hands.Hands()
mp_draw=mp.solutions.drawing_utils
sequences=deque(maxlen=20)

if not  cap.isOpened():
    print("Couldn't find a camera !")
    exit()

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("Couldn't catch a frame !")
        break

    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hands.process(rgb)

    if res.multi_hand_landmarks:
        for HANDS in res.multi_hand_landmarks:
            landmarks=[]
            for lm in HANDS.landmark:
                landmarks+=[lm.x,lm.y,lm.z]
            sequences.append(landmarks)

            mp_draw.draw_landmarks(frame,HANDS,mp.solutions.hands.HAND_CONNECTIONS)
    
    cv2.imshow("Video",frame)

    key=cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(sequences)==20:
            row=['r'] # r for right and we do the same again for left side
            for data in sequences:
                row+=data
            
            if len(row)==1261: # 1261= r + (((x,y,z)*21)*20)
                with open(CSV_FILE,"a",newline="") as file:
                    writer=csv.writer(file)
                    writer.writerow(row)
                
                sequences.clear()
                print("Saved !")
            
            else:
                print("Not all the 1261 input where saved !")
                break
        
        else:
            print("Not all the 20 frames where saved !")
            break
    
    elif key == ord('q'):
        print("End of recording !")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

