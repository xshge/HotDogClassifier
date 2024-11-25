

import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


st.header('Hot Dogs!')

uploaded_file = st.file_uploader("Pick a picture")
model = ResNet50(weights = 'imagenet')

if uploaded_file is not None:
    st.image(uploaded_file)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    results = decode_predictions(preds)
    st.write(results)

