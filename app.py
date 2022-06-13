import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary as summary_

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

st.title("CNN with PyTorch")

class CustomCNNModel(nn.Module) :
    def __init__(self) :
        super(CustomCNNModel, self).__init__()
        
        self.layer = nn.Sequential(
            # 8 * 28 * 28
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # 16 * 28 * 28
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 16 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 32 * 14 * 14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # 64 * 14 * 14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # 64 * 7 * 7
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fcLayer = nn.Sequential(
            nn.Linear(7 * 7 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x) :
        output = self.layer(x)
        output = output.view(output.size(0), -1)
        output = self.fcLayer(output)
        output = F.log_softmax(output, dim=1)
        return output
    
@st.cache
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomCNNModel().to(device)
    model.load_state_dict(torch.load("./model.pt", map_location=torch.device('cpu')))
    return model

def predict(img, model):
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    
    model.eval()
    tensor = torch.from_numpy(img).type(torch.FloatTensor)
    output = model(tensor)
    pred = output.argmax(dim=1, keepdim=True)
    return pred

model_load_state = st.text("Loading PyTorch Model...")
model = load_model()
model_load_state.text("PyTorch Model Loading Done! (using st.cache)")

st.subheader('Draw number!')
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=192,
    width=192,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Click button"):
    pred = predict(canvas_result.image_data, model)
    print(pred.tolist()[0][0])
    st.write(f'Probably, Your Number is {pred.tolist()[0][0]}')