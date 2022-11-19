import streamlit as st
import cv2
import pandas as pd
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
import keras
import tensorflow
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from keras.preprocessing import image
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras import layers
from keras.applications import EfficientNetB0



def check(img_path):
    
    #img_path = img_path.astype(np.uint8)
    loaded_model = tensorflow.keras.models.load_model("C:/Users/rashm/4.  BDA_sem3/rash.h5")
    img_path="C:/efficient_net/Dairy Milk/000001.jpg"
   # img_path=str(img_path)
    img1 = cv2.imread(img_path)
    img1 = cv2.resize(img1, (224, 224))
    x = np.expand_dims(img1, axis=0)
    x = preprocess_input(x)




    preds=loaded_model.predict(x)
    listt=preds.tolist()
    
    a=(max(listt))
    print("predicted class: ",a.index(max(a)))
    #print("predicted class: ",a)
    st.write(" ")
    st.subheader("Bill generator - Your Items are as follows:-\n")
    
    st.write("")  
    if a==0:
        st.write("Dairy Milk")
        amt=100
    elif a==1:
        st.write("Earphones")
        amt=500
    else:
        st.write("Shampoo")
        amt=300
    st.write("Total bill=",amt)
    st.write("  ")
    return ("Thank you, visit again")
    



st.title('Simpl.ify! - One stop for all needs')
st.subheader("An Automated checkout with inbuilt Face Recognizer")
#st.write("Say NO, to Queues anymore")
#st.image("D:/Down/dd.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
from PIL import Image
image = Image.open("D:/Down/dd.png")

st.image(image, caption='Say NO, to Queues anymore',output_format="auto",width=500,clamp=True)
face_cascade = cv2.CascadeClassifier("C:/Users/rashm/4.  BDA_sem3/haar.xml")

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("C:/Users/rashm/4.  BDA_sem3/trainingData.yml")
def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    name='Rashmi'
    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        id, uncertainty = rec.predict(gray[y:y + h, x:x + w])
        print(id, uncertainty)

        if (uncertainty< 53):
            if (math.ceil(id) == 45 or math.ceil(id) == 10 or math.ceil(id)== 22):
                name = "Rashmi"
                cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
        else:
            cv2.putText(img, 'name', (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
        #st.write(name)

    return img
def main():
    """Face Recognition App"""

    st.title("Let's get to know you.....")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition </h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    image_file=st.camera_input("Take a picture")
    if image_file is not None:
        our_image = Image.open(image_file)
        #st.text("Original Image")
        #st.image(our_image)

    if st.button("Recognise"):
        result_img= detect_faces(our_image)
        st.image(result_img)
        
    st.title("Let's scan your items")
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Item Recognition </h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    #img_path = st.text_input('Enter location of your image, '')
    img_path = st.file_uploader("Upload Image1", type=['jpg', 'png', 'jpeg'])
    if st.button("Predict"):
        st.write("Image Input is as follows")
        st.image(img_path)
        a=check(img_path)
        st.write(a)
        
    
    


if __name__ == '__main__':
    main()