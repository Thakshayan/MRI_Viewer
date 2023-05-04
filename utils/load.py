import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import os
import streamlit as st
from .Models.DD_Models.UNet import UNet as UNet2D
from .Models.DDD_Models.ModifiedUNet import MNet as MNet2D
from .Models.Gleason_Models.MNetWithBBClassifier import MNetWithBBClassifier
from .Models.DDD_Models.ModifiedUNet import MNet

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=( 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    #model = MNet().to(device)

    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(
        os.path.join(model_path),map_location=device))
    return model


@st.cache_resource
def load_model2D(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet2D().to(device)
    #model = MNet2D().to(device)
    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(
        os.path.join(model_path),map_location=device))

    return model

@st.cache_resource
def load_lesion_model(model_path):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MNet(4,1).to(device)

    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(
        os.path.join(model_path),map_location=device))

    return model

@st.cache_resource
def load_gleason_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MNetWithBBClassifier().to(device)

    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(
        os.path.join(model_path),map_location=device))
        print("Model Loaded")
    return model