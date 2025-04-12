import streamlit as st
from gradio_client import Client
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
import pytesseract
import cv2
import numpy as np
from google import genai
import textwrap
import base64
import io
import os
import subprocess

# Install Tesseract if not already installed
try:
    subprocess.run(["tesseract", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except (FileNotFoundError, subprocess.CalledProcessError):
    print("Tesseract not found. Installing Tesseract...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "tesseract-ocr"], check=True)


# Load the Base64-encoded font
def load_font_from_base64(encoded_font: str, size: int):
    font_bytes = base64.b64decode(encoded_font)
    font_stream = io.BytesIO(font_bytes)
    return ImageFont.truetype(font_stream, size)

# Read the Base64 font string from a file
with open("assets/fonts/font_base64.txt", "r") as font_file:
    encoded_font = font_file.read()

# Calculate center background brightness to decide font color
def get_background_brightness(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    crop_size = int(min(w, h) * 0.1)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - crop_size // 2, cy - crop_size // 2
    x2, y2 = cx + crop_size // 2, cy + crop_size // 2

    center_crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# Format text with line breaks for better layout
def format_text_with_linebreaks(text):
    punctuated = text.replace(". ", ".\n").replace("! ", "!\n").replace("? ", "?\n")
    return punctuated

# Draw centered, formatted corrected text on the image
def add_centered_text(image_path, text, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    img_width, img_height = pil_image.size

    font_size = 300
    avg_brightness = get_background_brightness(image_path)
    font_color = (0, 0, 0) if avg_brightness > 200 else (255, 255, 255)

    formatted_text = format_text_with_linebreaks(text)
    lines = []
    for sentence in formatted_text.split("\n"):
        lines.extend(textwrap.wrap(sentence.strip(), width=40))

    while True:
        font = load_font_from_base64(encoded_font, font_size)
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines]
        total_height = sum(line_heights) + 10 * (len(lines) - 1)
        max_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines])

        if max_width < img_width * 0.95 and total_height < img_height * 0.8:
            break
        font_size -= 2

    y = (img_height - total_height) // 2
    for line in lines:
        text_width = draw.textbbox((0, 0), line, font=font)[2]
        x = (img_width - text_width) // 2
        draw.text((x, y), line, font=font, fill=font_color)
        y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] + 10

    pil_image.save(output_path)

# OCR to extract words + detect their color
def extract_text_and_color(image_path):
    image = Image.open(image_path)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    cv_image = cv2.imread(image_path)
    text_boxes, font_colors = [], []

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        if word:
            x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
            text_region = cv_image[y:y+h, x:x+w]
            font_color = np.median(text_region.reshape(-1, 3), axis=0).astype(int) if text_region.size > 0 else (0, 0, 0)
            text_boxes.append((word, x, y, w, h))
            font_colors.append(tuple(font_color))

    return text_boxes, font_colors

# Streamlit UI
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Winky+Sans:ital,wght@0,300..900;1,300..900&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Winky Sans', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI-Powered Text-to-Image, Text Correction")
st.markdown("""
<div style="line-height: 1.2;">
    Here's a few examples to try:
    <ul>
        <li>Hello, World!</li>
        <li>This is a simple test.</li>
        <li>A whiteboard with text in it.</li>
        <li>A paper with text on it.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
user_text = st.text_area("Enter your text prompt below:", height=150)

if st.button("Generate Image"):
    if user_text:
        with open("input.txt", "w") as f:
            f.write(user_text)

        client = Client("black-forest-labs/FLUX.1-schnell")
        with st.spinner("Generating image... Please wait."):
            result = client.predict(prompt=user_text, seed=None, randomize_seed=True, width=512, height=512, num_inference_steps=20)

        if result:
            image_path = result[0]
            destination_path = "generated_image.webp"
            shutil.copy(image_path, destination_path)
            png_path = "generated_image.png"
            image = Image.open(destination_path)
            image.save(png_path, format="PNG")

            text_dimensions, font_colors = extract_text_and_color(png_path)
            if text_dimensions:
                ext_text = " ".join([word[0] for word in text_dimensions])
                try:
                    client_gemini = genai.Client(api_key="AIzaSyABRLypWnYp0nv3bCj1xl7ijv8V2WeaLsE")
                    response = client_gemini.models.generate_content(model="gemini-2.0-flash", contents=ext_text + " - Just provide me the corrected and meaningful text, nothing extra")
                    final_text_gemini = response.text.strip()
                except Exception as e:
                    st.error(f"Error in Gemini text correction: {e}")
                    final_text_gemini = ext_text

                image = cv2.imread(png_path)
                for i, (_, x, y, w, h) in enumerate(text_dimensions):
                    padding = 5
                    x1, y1 = max(x - padding, 0), max(y - padding, 0)
                    x2, y2 = min(x + w + padding, image.shape[1]), min(y + h + padding, image.shape[0])
                    bg_pixels = np.concatenate((
                        image[y1:y, x1:x2].reshape(-1, 3),
                        image[y+h:y2, x1:x2].reshape(-1, 3),
                        image[y1:y2, x1:x].reshape(-1, 3),
                        image[y1:y2, x+w:x2].reshape(-1, 3)
                    ))
                    bg_color = np.median(bg_pixels, axis=0).astype(int) if len(bg_pixels) > 0 else (255, 255, 255)
                    cv2.rectangle(image, (x, y), (x + w, y + h), bg_color.tolist(), -1)

                hidden_text_image_path = "hidden_text_image.png"
                cv2.imwrite(hidden_text_image_path, image)
                corrected_text_image_path = "output_image.png"
                add_centered_text(hidden_text_image_path, final_text_gemini, corrected_text_image_path)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(png_path, caption="Generated Image", use_container_width=True)
                with col2:
                    st.image(corrected_text_image_path, caption="Corrected Image", use_container_width=True)
            else:
                st.warning("No text was extracted from the image. Showing the original image.")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(png_path, caption="Generated Image", use_container_width=True)
                with col2:
                    st.image(png_path, caption="Generated Image (No text to correct)", use_container_width=True)
        else:
            st.error("No image was generated. Please try again.")
    else:
        st.warning("Please enter some text before generating an image.")

st.write("While this application is designed with simplicity in mind and may not handle all complex cases perfectly, it serves its purpose effectively for basic use.")
st.write("Feedback and contributions to improve it are always welcome! 🌟")
