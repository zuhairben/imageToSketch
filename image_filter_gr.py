import cv2
import numpy as np
from PIL import Image
import gradio as gr

def generate_sketch(image, is_colorized):
    """
    Generate a pencil sketch or colorized sketch from the given image.
    """
    if image is None:
        return None, "Failed to process the image!"

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

    return result, None

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
    return cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

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

def apply_filter(filter_option, input_image):
    """
    Apply the selected filter to the input image.
    """
    image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    if filter_option == "Grayscale Sketch":
        result, error = generate_sketch(image, False)
        if error:
            return None
    elif filter_option == "Colorized Sketch":
        result, error = generate_sketch(image, True)
        if error:
            return None
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
    else:
        result = image

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def main_interface():
    """
    Define the Gradio interface.
    """
    filter_options = ["Original", "Grayscale Sketch", "Colorized Sketch", "Sepia", "Enhanced", "Cartoon", "Edge Detection", "Blur"]

    interface = gr.Interface(
        fn=apply_filter,
        inputs=[gr.Dropdown(filter_options, label="Choose a Filter"), gr.Image(type="pil", label="Upload an Image")],
        outputs=gr.Image(type="pil", label="Filtered Image"),
        title="Image Filter Application",
        description="Apply various filters to your images!"
    )

    return interface

if __name__ == "__main__":
    main_interface().launch()
