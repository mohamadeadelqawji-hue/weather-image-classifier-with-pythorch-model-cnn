import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import Dataset,DataLoader;
import cv2
from glob import glob
import matplotlib.pyplot as plt
device="cuda"if torch.cuda.is_available() else "cpu";
train_url=r"C:\Users\User\Downloads\dataset2";

class Weather(Dataset):
  def __init__(self,folder):
    sunrise=glob(folder+"/sunrise/*");
    cloud=glob(folder+'/cloud/*');
    rain=glob(folder+'/rain/*');
    shine=glob(folder+'/shine/*');
    self.images=sunrise+cloud+rain+shine;
    self.fpaths=[0]*len(sunrise)+[1]*len(cloud)+[2]*len(rain)+[3]*len(shine);
  def __getitem__(self, index):
    targets =self.fpaths[index];
    image_f=self.images[index];
    image=cv2.imread(image_f)[:,:,::-1];
    image=cv2.resize(image,(224,224));
    return torch.tensor(image/255).float().permute(2,0,1),torch.tensor(targets).long()
  def __len__(self):
    return len(self.images);

def conv_layer(ni,no,kernel):
  return nn.Sequential(
    nn.Conv2d(ni,no,kernel,padding=kernel//2),
    nn.ReLU(),
    nn.BatchNorm2d(no),
    nn.MaxPool2d(2)
  )
def get_model():
  model=nn.Sequential(
    conv_layer(3,16,3),
    conv_layer(16,32,3),
    conv_layer(32,32,3),
    nn.Flatten(),
    nn.Linear(32 * 28 * 28, 128), 
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 4)
  ).to(device)
  loss_fn=nn.CrossEntropyLoss();
  optimizer=torch.optim.SGD(params=model.parameters(),lr=0.001);
  return model,loss_fn,optimizer;
def get_dataset():
  train=Weather(train_url);
  train_dl=DataLoader(train,batch_size=64,shuffle=True);
  return train_dl;
model,loss_fn,optimizer=get_model();

def model_train(x,y,model,loss_fn,optimizer):
  model.train();
  y_pred=model(x);
  loss_model=loss_fn(y_pred,y);
  optimizer.zero_grad();
  loss_model.backward();
  optimizer.step();
  return loss_model.item();


def train_epochs(train_dl, model, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = model_train(x_batch, y_batch, model, loss_fn, optimizer)
            total_loss += loss
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_dl):.4f}')

train_epochs(get_dataset(),model,loss_fn,optimizer)