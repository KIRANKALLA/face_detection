from PIL import Image
import numpy as np
import cv2
import streamlit as st
from retinaface import RetinaFace

st.header('AI&DS::RCEE')
st.title('No Of Faces Detector')

img = st.file_uploader('Take any picture')
if img:
    image = Image.open(img)
    st.image(image)
    image = np.array(image)
    obj = RetinaFace.detect_faces(image)
    for key in obj.keys():
    identity = obj[key]
    facial_area = identity['facial_area']
    cv2.rectangle(image,(facial_area[2],facial_area[3]),(facial_area[0],facial_area[1]),(0,0,0),5)
    st.image(image)
    
    
