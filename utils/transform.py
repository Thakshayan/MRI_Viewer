import numpy as np
from monai.transforms import Compose, AddChanneld,Orientationd,Resized,ToTensor, ScaleIntensityRanged,CropForegroundd, ToTensord
import torch

def transform(numpy_image, spatial_size):
    # Example numpy array with shape (H, W)

    pixdim =(1.5, 1.5, 1.0)
    a_min=0
    a_max=500

    transformations = Compose([
        lambda x: {"image": x, "label": None},
          AddChanneld(keys=["image"]),
          Orientationd(keys=["image"], axcodes="RAS"),
          ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
          CropForegroundd(keys=["image"], source_key="image"),
          Resized(keys=["image"], spatial_size=spatial_size),
          ToTensord(keys=["image"])
    ])

    # Apply the transformations to the numpy image
    transformed_tensor = transformations(numpy_image)
    #tensor_batch = transformed_tensor.unsqueeze(0)

    tensor = torch.unsqueeze(transformed_tensor['image'], 0)
    # Print the shape of the transformed tensor

    # output: torch.Size([1, 256, 256])
    return tensor

