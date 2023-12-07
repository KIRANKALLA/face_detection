import streamlit as st
import cv2
from retinaface import RetinaFace
import numpy as np
def main():
    st.title("Face Detection with RetinaFace")

    # Upload image through Streamlit
    image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if image is not None:
        # Convert Streamlit image to OpenCV format
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

        # Perform face detection using RetinaFace
        faces = RetinaFace.detect_faces(image)

        # Display the original image with bounding boxes around faces
        st.image(draw_boxes(image, faces), caption="Detected Faces", use_column_width=True)

def draw_boxes(image, faces):
    for face in faces:
        x, y, x2, y2 = face['x'], face['y'], face['x']+face['w'], face['y']+face['h']
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    return image

if __name__ == "__main__":
    main()
