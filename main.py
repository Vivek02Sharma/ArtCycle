import numpy as np
import streamlit as st
import tensorflow as tf
import PIL.Image as Image

import io
from huggingface_hub import hf_hub_download

# Heading
st.set_page_config(page_title = "ArtCycle",
                   page_icon = "./assets/favicon.png", layout = "centered")

st.markdown("<h1 style = 'text-align: center; color: #00d0ff;'>👨‍🎨ArtCycle🎨</h1>",
            unsafe_allow_html = True)

st.markdown("""
    ##### Overview
    The "ArtCycle" project uses advanced deep learning techniques to transform photos into paintings and vice versa, leveraging the power of CycleGAN (Generative Adversarial Networks). With this tool, users can upload a photo and see it converted into an artistic painting or take a painting and turn it into a photo. This offers a unique creative experience by combining art and technology in an intuitive and user-friendly interface.

    ##### CycleGAN
    CycleGAN is a deep learning model for image-to-image translation without paired data. It uses two generators and discriminators to convert images between domains, ensuring consistency through a cycle. This enables tasks like style transfer and image enhancement in an unsupervised manner.
""")

st.markdown("<p style = 'text-align: center; color: #00d0ff;'>Enjoy creating art with ArtCycle!</p>", unsafe_allow_html = True)

# Heading contents
col1, col2 = st.columns(2)
with col1:
    st.markdown("<p style = 'text-align: center; color: orange;'>Photo to Painting</p>",
            unsafe_allow_html = True)
with col2:
    st.markdown("<p style = 'text-align: center; color: orange;'>Painting to Photo</p>",
            unsafe_allow_html = True)

col1, col2 = st.columns(2)
with col1:
    st.image("./assets/photo2painting.png")
with col2:
    st.image("./assets/painting2photo.png")

#Left body
with st.sidebar:
    st.markdown('Use below sample images')

    col1, col2 = st.columns(2)
    with col1:
        st.text("Photos")
    with col2:
        st.text("Paintings")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo1.jpg")
    with col2:
        st.image(image = "./assets/paint1.jpg")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo2.jpg")
    with col2:
        st.image(image = "./assets/paint2.jpg")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo3.jpg")
    with col2:
        st.image(image = "./assets/paint3.jpg")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo4.jpg")
    with col2:
        st.image(image = "./assets/paint4.jpg")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo5.jpg")
    with col2:
        st.image(image = "./assets/paint5.jpg")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo6.jpg")
    with col2:
        st.image(image = "./assets/paint6.jpg")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo7.jpg")
    with col2:
        st.image(image = "./assets/paint7.jpg")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image = "./assets/photo8.jpg")
    with col2:
        st.image(image = "./assets/paint8.jpg")
    

# Middle body
col1, col2 = st.columns(2)
with col1:
    photo = st.file_uploader(
        "Photo2Painting"
    )
with col2:
    painting = st.file_uploader(
        "Painting2Photo"
    )

model_path_G = hf_hub_download(repo_id = "victor009/ArtCycle", filename = "gen_G_epoch_250.h5")
gen_G = tf.keras.models.load_model(model_path_G)

model_path_F = hf_hub_download(repo_id = "victor009/ArtCycle", filename = "gen_F_epoch_250.h5")
gen_F = tf.keras.models.load_model(model_path_F)


def load_image(image):
    image = Image.open(image)
    image = image.resize((256, 256))
    image = np.array(image) / 127.5 - 1
    return np.expand_dims(image, axis = 0)

def convert_image_back(image):
    image = (image + 1) * 127.5
    return np.array(image, dtype=np.uint8)

def save_image(image):
    img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format = 'PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

col1, col2 = st.columns(2)
with col1:
    if photo is not None:
        input_image = load_image(photo)
        if st.button('Convert to Painting'):
            output_image = gen_G.predict(input_image)
            output_image = convert_image_back(output_image[0])
            st.image(output_image, caption = "Converted Painting")

            img_byte_arr = save_image(output_image)
            st.download_button(
                label = "Download Converted Painting",
                data = img_byte_arr,
                file_name = "converted_painting.png",
                mime = "image/png"
            )
with col2:
    if painting is not None:
        input_image = load_image(painting)
        if st.button('Convert to Photo'):
            output_image = gen_F.predict(input_image)
            output_image = convert_image_back(output_image[0])
            st.image(output_image, caption = "Converted Photo")

            img_byte_arr = save_image(output_image)
            st.download_button(
                label = "Download Converted Photo",
                data = img_byte_arr,
                file_name = "converted_photo.png",
                mime = "image/png"
            )



# Footer part
st.markdown("""
    <footer style = "text-align: center; font-size: 12px; color: gray;">
        <p>ArtCycle | Created by Vivek Sharma</p>
        <p>For more details, visit the <a href = "https://huggingface.co/victor009/ArtCycle" target = "_blank">ArtCycle Hugging Face Model Page</a></p>
    </footer>
""", unsafe_allow_html = True)
