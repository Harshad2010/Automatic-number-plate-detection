import streamlit as st
from src.prediction.yolo_prediction_for_logo import YOLO_Pred1  
from src.prediction.logo_replacement import LogoReplacement
from PIL import Image
import numpy as np
import cv2


st.set_page_config(page_title="Replace License plate by Logo",
                   layout='wide',
                   page_icon='./icons/object.png')

st.header('Replace License plate by Logo')
st.write('Please Upload Image to get logo')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred1(onnx_model='src/model_training/yolov5/runs/train/Model2/weights/best.onnx',
                    data_yaml='src/model_training/yolov5/data.yaml')
    logo = LogoReplacement()
    #st.balloons()

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    logo_file = st.file_uploader(label='Logo Image')
    if image_file is not None:
        image_size_mb = image_file.size/(1024**2)
        file_details = {"filename": image_file.name,
                        "filetype": image_file.type,
                        "filesize": "{:,.2f} MB".format(image_size_mb)}
        #st.json(file_details)
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg')
            return {"file":image_file,
                    "logo_file":logo_file, 
                    "details":file_details}
        
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg, jpeg')
            return None
        
def main():
    object = upload_image()
      
    if object:
        prediction = False
        image_obj = Image.open(object['file'])
        logo_obj = Image.open(object['logo_file'])     
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj, logo_obj)
            
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Replace license plate by logo')
            if button:
                with st.spinner("""
                Replacing logo from image. please wait
                                """):
                    # below command will convert
                    # obj to array
                    image_array = np.array(image_obj)
                    logo_array = np.array(logo_obj)
                    pred_img = logo.replace_logo(image_array,logo_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
                
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5 model")
            st.image(pred_img_obj)
    
    
    
if __name__ == "__main__":
    main()


















