import numpy as np
from monai.transforms import Compose, AddChanneld,Spacingd,Orientationd,Resized,ToTensor, ScaleIntensityRanged,CropForegroundd, ToTensord
import torch

def transform(image_array, spatial_size=[128, 128, 16]):
    # Example numpy array with shape (H, W)

    pixdim =(1.5, 1.5, 1.0)
    a_min=0
    a_max=500

    transformation = Compose([
        lambda x: {"image": x},
          AddChanneld(keys=["image"]),
          Spacingd(keys=["image"], pixdim=pixdim),
          Orientationd(keys=["image"], axcodes="RAS"),
          ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0), 
          CropForegroundd(keys=["image"], source_key="image"),
          Resized(keys=["image"], spatial_size=spatial_size),
          ToTensord(keys=["image"])
    ])
    # Apply the transformations to the numpy image
    transformed_tensor = transformation(image_array)
    #tensor_batch = transformed_tensor.unsqueeze(0)
    tensor = torch.unsqueeze(transformed_tensor['image'], 0)
    # Print the shape of the transformed tensor

    
    return tensor

