import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

# Function to generate pencil sketch or colorized sketch
def generate_sketch(image_path, is_colorized, blur_ksize):
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
    blurred_image = cv2.GaussianBlur(inverted_image, (blur_ksize, blur_ksize), sigmaX=0, sigmaY=0)

    # Invert blurred image
    inverted_blurred = cv2.bitwise_not(blurred_image)

    # Create pencil sketch
    pencil_sketch_image = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    if is_colorized:
        result = cv2.addWeighted(image, 0.8, cv2.cvtColor(pencil_sketch_image, cv2.COLOR_GRAY2BGR), 0.2, 0)
    else:
        result = pencil_sketch_image

    return result

# Function for batch processing
def batch_process():
    folder_path = filedialog.askdirectory(title="Select Folder for Batch Processing")
    if not folder_path:
        return

    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        return

    is_colorized = colorized_var.get() == 1
    blur_ksize = slider.get()

    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        if not os.path.isfile(input_path):
            continue

        try:
            processed_image = generate_sketch(input_path, is_colorized, blur_ksize)
            if processed_image is not None:
                output_path = os.path.join(output_folder, f"processed_{filename}")
                cv2.imwrite(output_path, processed_image)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    messagebox.showinfo("Batch Processing", "Batch processing completed successfully!")

# Function to generate and display sketch with slider customization
def generate_and_display_sketch(is_colorized):
    if not image_path:
        messagebox.showerror("Error", "No image selected!")
        return

    blur_ksize = slider.get()  # Get slider value
    if blur_ksize % 2 == 0:  # Ensure blur kernel size is odd
        blur_ksize += 1

    sketch = generate_sketch(image_path, is_colorized, blur_ksize)
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

# Function to upload an image
def upload_image():
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

# Variables
image_path = ""
colorized_var = tk.IntVar()

# Widgets
lbl_title = tk.Label(root, text="Pencil Sketch Generator", font=("Arial", 16))
lbl_title.pack(pady=10)

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack()

frame_images = tk.Frame(root)
frame_images.pack(pady=10)

# Original image label
lbl_image = tk.Label(frame_images)
lbl_image.grid(row=0, column=0, padx=10)

# Filtered image label
lbl_result = tk.Label(frame_images)
lbl_result.grid(row=0, column=1, padx=10)

# Options
frame_options = tk.Frame(root)
frame_options.pack(pady=10)

chk_colorized = tk.Checkbutton(frame_options, text="Colorized Sketch", variable=colorized_var)
chk_colorized.grid(row=0, column=0, padx=10)

slider = tk.Scale(frame_options, from_=1, to=51, resolution=2, orient="horizontal", label="Blur Kernel Size")
slider.set(21)
slider.grid(row=0, column=1, padx=10)

btn_generate = tk.Button(frame_options, text="Generate Sketch", 
                         command=lambda: generate_and_display_sketch(colorized_var.get() == 1))
btn_generate.grid(row=0, column=2, padx=10)

# Batch Processing Button
btn_batch = tk.Button(root, text="Batch Process Images", command=batch_process)
btn_batch.pack(pady=10)

# Run GUI
root.mainloop()
