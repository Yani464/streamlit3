import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from io import BytesIO
import numpy as np
from model import Muffler,UNet
import cv2

def page1():
    st.title("Коррекция шумов изображения с помощью автоинкодера")


    def load_model_auto():
        loaded_model = Muffler()
        loaded_model.load_state_dict(torch.load('autoencoder_model.pth',map_location=torch.device('cpu')))
        loaded_model.eval()
        return loaded_model

    model = load_model_auto()

    def inference(image):
        transform = transforms.Compose([
            transforms.Resize((540, 420)),  # Изменение размера до 420x540
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output_image = model(image)
        
        output_image = output_image.squeeze(0).squeeze(0).cpu().numpy()

        return output_image

    # st.title("Использование модели для обработки изображений")

    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)

        input_image = Image.open(uploaded_image)
        processed_image = inference(input_image)

        st.image(processed_image, caption="Обработанное изображение", use_column_width=True, channels="GRAY")


def page2():
    st.title("Определение кораблей с космических снимков при помощи модели YOLOv5")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

    image = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

    if image:
        image = Image.open(image)
        image = np.array(image)

        results = model(image)

        st.image(results.render()[0], use_column_width=True)

        st.write("Обнаруженные объекты:")
        for obj in results.xyxy[0].numpy():
            label = int(obj[5])
            confidence = obj[4]
            st.write(f"Класс: Корабли, Уверенность: {confidence:.2f}")

def page3():
    st.title("Unet")
    
    def load_model():
        loaded_model_unet = UNet()  # Replace with the correct initialization for your UNet model
        loaded_model_unet.load_state_dict(torch.load('unet_model.pth'))
        loaded_model_unet.eval()
        return loaded_model_unet

    model_unet = load_model()

    def inference(image):
        transform = transforms.Compose([
            transforms.Resize((250, 250)),  # Resize to 250x250
            transforms.ToTensor()
        ])
        
        # Ensure the image has 3 channels (RGB)
        image = image.convert('RGB')
        
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output_image = model_unet(image)
        
        # Normalize the output image to be in the [0, 255] range
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
        output_image = output_image.squeeze(0).cpu().numpy()  # Remove the batch dimension and convert to numpy
        
        # Convert to single channel grayscale image
        output_image = output_image.mean(axis=0).astype('uint8')  # Take mean across channels

        return output_image

    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)
        
        input_image = Image.open(uploaded_image)
        processed_image = inference(input_image)
        
        # Display the processed image as grayscale
        st.image(processed_image, caption="Обработанное изображение", use_column_width=True, channels="GRAY")


if 'page' not in st.session_state:
    st.session_state.page = 'Коррекция шумов'

def main():
    st.sidebar.title("Меню")
    page_selection = st.sidebar.radio("Выберите страницу", ["Коррекция шумов", "Определение кораблей", "Unet"])

    # Handle page switching and reset state
    if page_selection != st.session_state.page:
        st.session_state.page = page_selection
        st.experimental_rerun()  # Reset the page

    if st.session_state.page == 'Коррекция шумов':
        page1()
    elif st.session_state.page == 'Определение кораблей':
        page2()
    elif st.session_state.page == 'Unet':
        page3()

if __name__ == "__main__":
    main()

