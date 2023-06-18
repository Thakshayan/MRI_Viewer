import streamlit as st
import os
import nibabel as nib
import numpy as np

from streamlit import session_state
from streamlit_option_menu import option_menu
from utils.view import plot_slice, plot_image_label, plot_2D_image_label
from utils.load import load_model, load_model2D, load_lesion_model, load_gleason_model
from utils.transform import transform, lesion_transform, remove_slices

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
        icons=['house', 'eye-fill','','badge-3d-fill'], menu_icon="cast",default_index=0,
          styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": { "font-size": "25px"}, 
            "nav-link": { "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {},
            "menu-icon": {"display":"none","font-size": "25px"},
            "menu-title" : {"font-size": "25px"}
        }
        )
    
st.sidebar.success("")

# # horizontal Menu
# selected = option_menu(None, ["Home", 'View','2D Model', '3D Model'], 
#     icons=['house', 'eye-fill','','badge-3d-fill'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")


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

if selected == '2D Model':
    selectedZone = option_menu(None, ["Prostate", 'Pheripheral','Transition'], 
            icons=['', '',''], 
            menu_icon="cast", default_index=0, orientation="horizontal"
            )
    
    st.title("Predicted By 2D Model")
    if 'file_path' in session_state and session_state.file_path:
        # # horizontal Menu
       
        if selectedZone == 'Prostate':
            model_path = './saved_models/2D_Models/prostate/best_metric_model.pth'
        if selectedZone == 'Pheripheral':
            model_path = './saved_models/2D_Models/pheripheral/best_metric_model.pth'
        if selectedZone == 'Transition':
            model_path = './saved_models/2D_Models/transition/best_metric_model.pth'
        model = load_model2D(model_path)
        nifti_image = nib.load(session_state.file_path)
        image_data = nifti_image.get_fdata()

        total_slize_size = image_data.shape[2]
        image_slices = np.split(image_data, total_slize_size, axis=2)
        
        labels = []
        images = []
        spatial_size = [128, 128]
        for i in range(total_slize_size):
            newimg = image_slices[i].squeeze() 
            
            transformed_image = transform(newimg, spatial_size)
            
            label = model(transformed_image)
            images.append(transformed_image)
            labels.append(label)
        plot_2D_image_label(images, labels)
            
    else:
        st.write("Yet to upload a file")

if selected == '3D Model':
    
    selectedZone = option_menu(None, ["Prostate", 'Pheripheral','Transition','Lesion'], 
            icons=['', '','','', ''], 
            menu_icon="cast", default_index=0, orientation="horizontal"
            )
    
    st.title("Predicted By 3D Model")
    if 'file_path' in session_state and session_state.file_path:
        # # horizontal Menu
        if selectedZone != 'Lesion':
            if selectedZone == 'Prostate':
                model_path = './saved_models/3D_Models/prostate/best_metric_model.pth'
            if selectedZone == 'Pheripheral':
                model_path = './saved_models/3D_Models/pheripheral/best_metric_model.pth'
            if selectedZone == 'Transition':
                model_path = './saved_models/3D_Models/transition/best_metric_model.pth'
        
        
            model = load_model(model_path)
            nifti_image = nib.load(session_state.file_path)
            if nifti_image.shape[2] > 16:
                nifti_image = remove_slices(nifti_image)
            image_data = nifti_image.get_fdata()
            spatial_size = [128, 128, 16]
            transformed_image = transform(image_data, spatial_size)
            
            label = model(transformed_image)
            
            session_state.labeled = True
            plot_image_label(transformed_image, label)
        else:
            lesion_model_path = './saved_models/3D_Models/lesion/best_metric_model.pth'
            Gleason_model_path = './saved_models/3D_Models/gleason/best_metric_model.pth'
            prostate_model_path = './saved_models/3D_Models/prostate/best_metric_model.pth'
            pheripheral_model_path = './saved_models/3D_Models/pheripheral/best_metric_model.pth'
            transition_model_path =  './saved_models/3D_Models/transition/best_metric_model.pth'
        

            lesion_model = load_lesion_model(lesion_model_path)
            gleason_model = load_gleason_model(Gleason_model_path)
            prostate_model = load_model(prostate_model_path)
            transition_model = load_model(pheripheral_model_path)
            pheripheral_model = load_model(transition_model_path)

            nifti_image = nib.load(session_state.file_path)
            image_data = nifti_image.get_fdata()
            spatial_size = [128, 128, 16]
            transformed_image = transform(image_data, spatial_size)
            
            prostate_label = prostate_model(transformed_image)
            transition_label = transition_model(transformed_image)
            pheripheral_label = pheripheral_model(transformed_image)

            image_array = {
                "image" : transformed_image, 
                "PZ" : pheripheral_label, 
                "TZ" : transition_label, 
                "prostate" : prostate_label
            }
           
            
            transform_lesion_image = lesion_transform(image_array)

            lesion_label = lesion_model(transform_lesion_image)
            gleason_score = gleason_model(transform_lesion_image)

            score = np.argmax(gleason_score)


            if score < 1 :
                st.success(f"Predicted Gleason Score: {score}")
                st.success(f"Predicted Gleason Score Accuray: {gleason_score[0][score].detach().numpy():.4f}")
                st.success("Indicate No Biopsy Needed")
            elif score ==2:
                st.error(f"Predicted Gleason Score: {score}")
                st.success(f"Predicted Gleason Score Accuray: {gleason_score[0][score].detach().numpy():.4f}")
                st.success("Indicate intermediate")
            else:
                st.error(f"Predicted Gleason Score: {score}")
                st.success(f"Predicted Gleason Score Accuray: {gleason_score[0][score].detach().numpy():.4f}")
                if score == 3:
                    st.error("Indicate Biopsy Needed (Intermediate Unfavorable)")
                else:
                    st.error("Indicate Biopsy Needed (High Risk)")

            session_state.labeled = True
            plot_image_label(transformed_image, lesion_label)
            
    else:
        st.write("Yet to upload a file")