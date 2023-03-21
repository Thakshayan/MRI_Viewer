import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import math

@st.cache_data
def plot_slice(image, slice):
    
    slice_data = image[:,:,slice]
    # Create a figure and axes
    fig, ax = plt.subplots()

    plt.axis('off')

    # Display the 3D numpy array as an image
    ax.imshow(slice_data, cmap='gray')

    # Display the figure in Streamlit using st.pyplot()
    st.pyplot(fig)



def plot_image_label(image, label):
    label_data = label.get_array()
    slider_value = st.slider('Select a value', min_value=0, max_value=image.shape[4] - 1,
                         value= image.shape[4] //2, key='2')
    slice_image = image[0,0,:,:,slider_value]
    slice_label = label_data[0,0,:,:,slider_value]
    # session_state.labeled = True
    fig, ax = plt.subplots()

    # Display MRI image
    ax.imshow(slice_image, cmap='gray', alpha=1)

    # Set alpha channel of label image
    alpha = np.zeros_like(slice_label)
    alpha[slice_label > 0.5] = 0.4

    # Display label image
    ax.imshow(slice_label, cmap='jet', alpha=alpha)

    # Add colorbar for label image
    #cbar = plt.colorbar(ax.imshow(label, cmap='jet', alpha=1))

    # Set title for plot
    ax.set_title('MRI Image with Label')

    # Display the figure in Streamlit using st.pyplot()
    st.pyplot(fig)

def plot_2D_image_label(images, labels):
    ncols = 3
    nrows = math.ceil(len(images)/3)
    fig, axs = plt.subplots(nrows= nrows, ncols= ncols, figsize=(4 * ncols, 4 * nrows),
                        gridspec_kw={'width_ratios': [1] * ncols, 'height_ratios': [1] * nrows})
    # Iterate through the images and plot them in subplots
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            # Set alpha channel of label image
            image = images[i][0,0,:,:]
            label = labels[i][0,0,:,:].get_array()

            # Display MRI image
            ax.imshow(image, cmap='gray', alpha=1)

            alpha = np.zeros_like(label)
            alpha[label > 0.5] = 0.5
            
            ax.imshow(label, cmap='jet', alpha=alpha)
        ax.axis('off')  # turn off axis labels
        
    plt.tight_layout()  # adjust subplot spacing
    st.pyplot(fig)