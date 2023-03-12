import streamlit as st
import os
import nibabel as nib

from streamlit import session_state
from streamlit_option_menu import option_menu
from utils.view import plot_slice

st.set_page_config(
    page_title="Doc ASk",
    page_icon="👨‍⚕️",
)



# Define a directory to save the uploaded NIfTI files
UPLOAD_DIRECTORY = "./images"


# Create the directory if it doesn't already exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


with st.sidebar:
    selected = option_menu(None, ["Home", 'View','2D Model', '3D Model'], 
        icons=['house', 'eye-fill','','badge-3d-fill'], menu_icon="cast", default_index=0)
    selected
st.sidebar.success("")

if selected == 'Home':   
    # Add a file uploader widget to the Streamlit app
    
    st.title("Home Page")
    nifti_file = st.file_uploader("Choose image to evaluate model", type=["nii", "nii.gz"], key='file')
    st.write("Click Submit")
    submit = st.button("Submit")

    if submit and nifti_file :
        session_state.clear()
        file_path = os.path.join(UPLOAD_DIRECTORY, nifti_file.name )
        
        with open(file_path, "wb") as file:
            file.write(nifti_file.getbuffer())
            st.write("Saved file:", file_path)
        # Create a session state object
        st.success("Successfully uploaded")
        session_state.file_path = file_path
        session_state.name = nifti_file.name

    if 'name' in session_state and session_state.name:
        st.success("Currently viewing ")

if selected == 'View': 
    st.title("View Page")
    if 'file_path' in session_state and session_state.file_path:
        nifti_image = nib.load(session_state.file_path)
        image_data = nifti_image.get_fdata()
        st.text(image_data.shape)
        slider_value = st.slider('Select a value', min_value=0, max_value=image_data.shape[2] - 1,
                            value= image_data.shape[2] //2, key='1')
        
        plot_slice(image_data,slider_value)
    else:
        st.write("Yet to upload a file")

if selected == '3D':
    st.write("")