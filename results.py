import torch
import torch.nn as nn
import cv2
import mediapipe as mp
from collections import deque
import subprocess
import time

FRAMES=20
FEATUREs=63

HIDDEN_SIZE=128
NUM_LAYERS=2

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru=nn.GRU(
            input_size=FEATUREs,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3
        )

        self.fc=nn.Linear(HIDDEN_SIZE,2)

    def forward(self,x):
        out,_=self.gru(x)
        out=out[:, -1 ,:]
        out=self.fc(out)
        return out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=GRUModel()
model.load_state_dict(torch.load("gru_model.pth",map_location=device))
model.eval()

cap=cv2.VideoCapture(0)
hands=mp.solutions.hands.Hands()
mp_draw=mp.solutions.drawing_utils
seqences=deque(maxlen=20)

last_action_time=0
cool_time=1.5
label_map={0:'r',1:'l'}

if not cap.isOpened():
    print("Couldn't find the camera !")
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
            seqences.append(landmarks)

            mp_draw.draw_landmarks(frame,HANDS,mp.solutions.hands.HAND_CONNECTIONS)
        
        if len(seqences)==FRAMES:

            X=torch.tensor([list(seqences)], dtype=torch.float32).to(device)

            with torch.no_grad():
                output=model(X)
                prediction=torch.argmax(output,dim=1).item()
                label=label_map[prediction]
            
            
            now=time.time()

            if now - last_action_time > cool_time:
                if label =='r':
                    
                    subprocess.run(['xdotool', 'windowfocus', '54525956', 'key', 'Right'])
                 
                elif label == 'l':
                    
                    subprocess.run(['xdotool', 'windowfocus', '54525956', 'key', 'Left'])
                
                last_action_time=now
        
        
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()

