import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


def generate_sketch(image_path, is_colorized):
    """
    Generate a pencil sketch or colorized sketch from the given image.
    """
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to upload the image!")
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
    sepia_image = np.clip(sepia_image, 0, 255)  # Ensure pixel values are valid
    return sepia_image.astype('uint8')

def generate_enhance(image):
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return enhanced

def generate_cartoon(image):
    """
    Apply a cartoon effect to the given image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    gray = cv2.medianBlur(gray, 7)
    # Detect edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Apply bilateral filter for cartoon effect
    color = cv2.bilateralFilter(image, 9, 250, 250)
    # Combine edges and color
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

def generate_and_display_gallery():
    """
    Generate and display a preview gallery of all filters.
    """
    if not image_path:
        messagebox.showerror("Error", "No image selected!")
        return
    
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

    # Display gallery
    gallery_window = tk.Toplevel(root)
    gallery_window.title("Filter Gallery")
    gallery_window.geometry("800x600")

    for i, (effect_name, effect_image) in enumerate(effects.items()):
        effect_bgr = cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB) if effect_name != "Grayscale Sketch" else effect_image
        img = Image.fromarray(effect_bgr)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        
        lbl_effect = tk.Label(gallery_window, text=effect_name, font=("Arial", 10))
        lbl_effect.grid(row=i // 4, column=(i % 4) * 2, pady=10, padx=10)
        
        lbl_preview = tk.Label(gallery_window, image=img_tk)
        lbl_preview.image = img_tk
        lbl_preview.grid(row=i // 4, column=(i % 4) * 2 + 1, pady=10, padx=10)
        
        # Add click event to save the selected image
        lbl_preview.bind("<Button-1>", lambda event, img=effect_image: save_gallery_image(img))

def generate_and_display_sketch(is_colorized):
    """
    Generate and display a pencil sketch or colorized sketch.
    """
    if not image_path:
        messagebox.showerror("Error", "No image selected!")
        return

    sketch = generate_sketch(image_path, is_colorized)
    if sketch is not None:
        if not is_colorized:
            sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        else:
            sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(sketch_bgr)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        lbl_result.config(image=img)
        lbl_result.image = img

        # Enable save button
        # btn_save.config(state="normal")
        # btn_save.config(command=lambda: save_output(sketch))



def generate_and_display_effect(effect_name):
    """
    Generate and display a specific filter effect (e.g., Sepia).
    """
    if not image_path:
        messagebox.showerror("Error", "No image selected!")
        return

    image = cv2.imread(image_path)
    if effect_name == "sepia":
        result = generate_sepia(image)
    elif effect_name == "enhance":
        result = generate_enhance(image)
    elif effect_name == "cartoon":
        result = generate_cartoon(image)
    elif effect_name == "edge":
        result = generate_edge_detection(image)
    elif effect_name == "blur":
        result = generate_blur(image)
    else:
        messagebox.showerror("Error", "Effect not implemented!")
        return

    result_bgr = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(result_bgr)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    lbl_result.config(image=img)
    lbl_result.image = img

    # Enable save button
    # btn_save.config(state="normal")
    # btn_save.config(command=lambda: save_output(result))

def save_gallery_image(image):
    """
    Save the image from the gallery.
    """
    output_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                               filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
    if output_path:
        cv2.imwrite(output_path, image)
        messagebox.showinfo("Success", "Image saved successfully!")


def upload_image():
    """
    Upload an image to process.
    """
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if image_path:
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        lbl_image.config(image=img)
        lbl_image.image = img




# GUI Setup
root = tk.Tk()
root.title("Pencil Sketch Generator")

# Widgets
image_path = ""
lbl_title = tk.Label(root, text="Pencil Sketch Generator", font=("Arial", 16))
lbl_title.pack(pady=10)

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack()

lbl_image = tk.Label(root)
lbl_image.pack(pady=10)

btn_grayscale = tk.Button(root, text="Generate Grayscale Sketch", command=lambda: generate_and_display_sketch(False))
btn_grayscale.pack(pady=5)

btn_colorized = tk.Button(root, text="Generate Colorized Sketch", command=lambda: generate_and_display_sketch(True))
btn_colorized.pack(pady=5)

btn_sepia = tk.Button(root, text="Generate Sepia Filter", command=lambda: generate_and_display_effect("sepia"))
btn_sepia.pack(pady=5)

btn_sepia = tk.Button(root, text="Generate Enhance Filter", command=lambda: generate_and_display_effect("enhance"))
btn_sepia.pack(pady=5)

btn_cartoon = tk.Button(root, text="Generate Cartoon Effect", command=lambda: generate_and_display_effect("cartoon"))
btn_cartoon.pack(pady=5)

btn_edge = tk.Button(root, text="Generate Edge Detection", command=lambda: generate_and_display_effect("edge"))
btn_edge.pack(pady=5)

btn_blur = tk.Button(root, text="Generate Blur", command=lambda: generate_and_display_effect("blur"))
btn_blur.pack(pady=5)

btn_gallery = tk.Button(root, text="Preview All Filters", command=generate_and_display_gallery)
btn_gallery.pack(pady=10)

lbl_result = tk.Label(root)
lbl_result.pack(pady=10)

# Run the GUI
root.mainloop()
