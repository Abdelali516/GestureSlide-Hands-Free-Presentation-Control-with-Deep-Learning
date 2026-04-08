import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

CSV_FILE="/home/abdelali/Downloads/dataset.csv"
FRAMES=20
FEATURES=63

HIDDEN_SIZE=128
BATCH=32
EPOCHS=50

NUM_LAYERS=2
LEARNING_RATE=0.001

df=pd.read_csv(CSV_FILE,header=None)

y=df.iloc[:, 0].values # the labels ["r","l"]
X=df.iloc[:, 1:].values # the rest

label_map={'r':0,'l':1}
y=np.array([ label_map[label] for label in y])

X=X.reshape(-1,FRAMES,FEATURES)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)

train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)

train_loader=DataLoader(train_data,batch_size=BATCH,shuffle=True)
test_loader=DataLoader(test_data,batch_size=BATCH,shuffle=False)

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru=nn.GRU(
            input_size=FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.3
        )

        self.fc=nn.Linear(HIDDEN_SIZE,2) # 2 for 'r' & 'l'

    def forward(self,x):
        out,_=self.gru(x)
        out=out[:, -1 ,:]
        out=self.fc(out)
        return out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=GRUModel()

criteria=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):
    model.train()
    total_loss=0

    for X_batch,y_batch in train_loader:
        X_batch,y_batch=X_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        prediction=model(X_batch)
        loss=criteria(prediction,y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    
    model.train()
    correct=0
    total=0

    with torch.no_grad():
        for X_batch,y_batch in test_loader:
            X_batch,y_batch=X_batch.to(device),y_batch.to(device)
            prediction=model(X_batch)
            predicted_sign=torch.argmax(prediction,dim=1)
            correct+=(predicted_sign==y_batch).sum().item()
            total+=y_batch.size(0)
        
    acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    print(f" Epoch = {epoch +1 }/{EPOCHS} | Loss : {avg_loss:.4f} | Accuracy: {acc:.2f}%")

torch.save(model.state_dict(),"gru_model.pth")
print("Model saved !")