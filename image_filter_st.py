import cv2
import numpy as np
import streamlit as st
from PIL import Image


def generate_sketch(image_path, is_colorized):
    """
    Generate a pencil sketch or colorized sketch from the given image.
    """
    image = cv2.imread(image_path)
    if image is None:
        st.error("Failed to upload the image!")
        return

    # Denoising
    if is_colorized:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert grayscale image
    inverted_image = cv2.bitwise_not(gray_image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(inverted_image, (51, 51), sigmaX=0, sigmaY=0)

    # Invert blurred image
    inverted_blurred = cv2.bitwise_not(blurred_image)

    # Create pencil sketch
    pencil_sketch_image = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    if is_colorized:
        result = cv2.addWeighted(image, 0.8, cv2.cvtColor(pencil_sketch_image, cv2.COLOR_GRAY2BGR), 0.2, 0)
    else:
        result = pencil_sketch_image

    return result

def generate_sepia(image):
    """
    Apply a sepia filter to the given image.
    """
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image.astype('uint8')

def generate_enhance(image):
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return enhanced

def generate_cartoon(image):
    """
    Apply a cartoon effect to the given image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def generate_edge_detection(image):
    """
    Apply edge detection to the given image.
    """
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def generate_blur(image):
    """
    Apply Gaussian blur to the given image.
    """
    return cv2.GaussianBlur(image, (15, 15), 0)

def display_gallery(image_path):
    """
    Display a gallery of all filters.
    """
    image = cv2.imread(image_path)
    effects = {
        "Original": image,
        "Grayscale Sketch": generate_sketch(image_path, False),
        "Colorized Sketch": generate_sketch(image_path, True),
        "Sepia": generate_sepia(image),
        "Enhanced": generate_enhance(image),
        "Cartoon": generate_cartoon(image),
        "Edge Detection": generate_edge_detection(image),
        "Blur": generate_blur(image),
    }

    st.subheader("Filter Gallery")
    cols = st.columns(4)
    for i, (effect_name, effect_image) in enumerate(effects.items()):
        effect_bgr = cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB) if effect_name != "Grayscale Sketch" else effect_image
        img = Image.fromarray(effect_bgr)
        with cols[i % 4]:
            st.image(img, caption=effect_name, use_container_width=True)

# Streamlit app
st.title("Image Filter Application")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(file_path), caption="Uploaded Image", use_container_width=True)

    filter_option = st.selectbox("Choose a Filter", ["None", "Grayscale Sketch", "Colorized Sketch", "Sepia", "Enhanced", "Cartoon", "Edge Detection", "Blur", "Gallery"])

    if filter_option != "None":
        image = cv2.imread(file_path)

        if filter_option == "Grayscale Sketch":
            result = generate_sketch(file_path, False)
        elif filter_option == "Colorized Sketch":
            result = generate_sketch(file_path, True)
        elif filter_option == "Sepia":
            result = generate_sepia(image)
        elif filter_option == "Enhanced":
            result = generate_enhance(image)
        elif filter_option == "Cartoon":
            result = generate_cartoon(image)
        elif filter_option == "Edge Detection":
            result = generate_edge_detection(image)
        elif filter_option == "Blur":
            result = generate_blur(image)
        elif filter_option == "Gallery":
            display_gallery(file_path)
            result = None

        if result is not None:
            st.image(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), caption="Filtered Image", use_container_width=True)

            if st.button("Download Image"):
                output_path = "output.jpg"
                cv2.imwrite(output_path, result)
                with open(output_path, "rb") as file:
                    st.download_button(label="Download", data=file, file_name="filtered_image.jpg", mime="image/jpeg")
