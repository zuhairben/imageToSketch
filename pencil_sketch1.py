import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
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
    # Step 1: Edge Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, blockSize=9, C=9)

    # Step 2: Smoothing with bilateral filter
    color = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Color Quantization using k-means clustering
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)
    K = 9  # Number of clusters (colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(color.shape)

    # Step 4: Combine edges with color image
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

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

    effect_images = []  # To store effect images for saving
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
        
        # Add click event to save individual image
        lbl_preview.bind("<Button-1>", lambda event, img=effect_image: save_gallery_image(img))
        
        # Append image and name for saving all
        effect_images.append((effect_name, effect_image))

    # Add a "Save All" button at the bottom
    btn_save_all = tk.Button(
        gallery_window, 
        text="Save All Filters", 
        command=lambda: save_gallery_image(effect_images)
    )
    btn_save_all.grid(row=(len(effects) // 4) + 1, column=0, columnspan=8, pady=20)

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



def save_gallery_image(image):
    folder_path = filedialog.askdirectory(title="Select Folder to Save All Filters")
    if not folder_path:
        return  # User canceled folder selection

    # Ask the user for a name for the folder to create
    folder_name = simpledialog.askstring("Folder Name", "Enter folder name to save images:")
    if not folder_name:
        return  # User canceled the name input
    
    full_path = os.path.join(folder_path, folder_name)
    os.makedirs(full_path, exist_ok=True)  # Create the folder

    # Save all images in the created folder
    for effect_name, effect_image in image:
        file_name = f"{effect_name.replace(' ', '_')}.jpg"  # Replace spaces with underscores
        output_path = os.path.join(full_path, file_name)
        cv2.imwrite(output_path, effect_image)

    messagebox.showinfo("Success", f"All images saved successfully in {full_path}!")




def save_filtered_image(image):
    if image is None:
        messagebox.showerror("Error", "No filtered image to save!")
        return
    
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

def apply_filter():
    global current_filtered_image  # Ensure this is updated
    if not image_path:
        messagebox.showerror("Error", "No image selected!")
        return

    selected_filter = filter_combobox.get()
    if not selected_filter:
        messagebox.showerror("Error", "No filter selected!")
        return

    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image! Please upload a valid image.")
        return

    # Apply the selected filter
    if selected_filter == "Grayscale Sketch":
        current_filtered_image = generate_sketch(image_path, False)
    elif selected_filter == "Colorized Sketch":
        current_filtered_image = generate_sketch(image_path, True)
    elif selected_filter == "Sepia":
        current_filtered_image = generate_sepia(image)
    elif selected_filter == "Enhanced":
        current_filtered_image = generate_enhance(image)
    elif selected_filter == "Cartoon":
        current_filtered_image = generate_cartoon(image)
    elif selected_filter == "Edge Detection":
        current_filtered_image = generate_edge_detection(image)
    elif selected_filter == "Blur":
        current_filtered_image = generate_blur(image)
    else:
        messagebox.showerror("Error", "Selected filter is not implemented!")
        return

    # Display the filtered image
    if selected_filter in ["Grayscale Sketch", "Colorized Sketch"]:
        result_bgr = cv2.cvtColor(current_filtered_image, cv2.COLOR_GRAY2BGR if selected_filter == "Grayscale Sketch" else cv2.COLOR_BGR2RGB)
    else:
        result_bgr = cv2.cvtColor(current_filtered_image, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(result_bgr)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    lbl_result.config(image=img)
    lbl_result.image = img

    # Enable the save button
    btn_save.config(state="normal", command=lambda: save_filtered_image(current_filtered_image))



# GUI Setup
root = tk.Tk()
root.title("Pencil Sketch Generator")

# Widgets
image_path = ""
current_filtered_image = None  # To store the currently displayed filtered image
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

# Dropdown menu for filters
lbl_filter = tk.Label(root, text="Select a Filter:", font=("Arial", 12))
lbl_filter.pack(pady=5)

filter_combobox = ttk.Combobox(root, values=[
    "Grayscale Sketch",
    "Colorized Sketch",
    "Sepia",
    "Enhanced",
    "Cartoon",
    "Edge Detection",
    "Blur"
], state="readonly")
filter_combobox.pack()

btn_apply_filter = tk.Button(root, text="Apply Filter", command=apply_filter)
btn_apply_filter.pack(pady=10)

btn_gallery = tk.Button(root, text="Preview All Filters", command=generate_and_display_gallery)
btn_gallery.pack(pady=10)

# Save button for the filtered image
btn_save = tk.Button(root, text="Save Filtered Image", command=lambda: save_gallery_image(lbl_result.image))
btn_save.pack(pady=10)

# Run the GUI
root.mainloop()
