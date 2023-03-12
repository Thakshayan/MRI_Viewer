import matplotlib.pyplot as plt
import streamlit as st


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