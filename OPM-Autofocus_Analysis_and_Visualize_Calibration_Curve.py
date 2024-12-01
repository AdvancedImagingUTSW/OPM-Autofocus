# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:33:33 2024

@author: S233755
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def select_images():
    """Opens a dialog for selecting image files."""
    files = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[("All Files", "*.*"), ("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")]
    )
    if files:
        normalized_files = [os.path.normpath(file) for file in files]
        image_files.set(";".join(normalized_files))
        selected_files_label.config(text=f"{len(files)} files selected")


def select_output_folder():
    """Opens a dialog for selecting the output folder."""
    folder = filedialog.askdirectory(title="Select Output Folder")
    if folder:
        output_folder.set(os.path.normpath(folder))


def reset_gui():
    """Resets the video and MIP display when colormap or settings change."""
    global colored_images_stack, mip_array, video_playing
    colored_images_stack = []
    mip_array = None
    video_playing = False
    stop_video()
    clear_placeholder(mip_label)
    clear_placeholder(video_label)


def clear_placeholder(label):
    """Clears the placeholder image from a label."""
    placeholder = Image.new("RGB", (300, 300), "black")
    placeholder_tk = ImageTk.PhotoImage(placeholder)
    label.config(image=placeholder_tk)
    label.image = placeholder_tk


def apply_temporal_color_coding():
    """Applies temporal color coding, saves results, and displays MIP."""
    reset_gui()
    files = image_files.get().split(";")
    folder = output_folder.get()

    if not files or not folder:
        messagebox.showerror("Error", "Please select image files and an output folder.")
        return

    try:
        # Get selected colormap
        selected_colormap_name = colormap_var.get()
        selected_colormap = cm.get_cmap(selected_colormap_name)

        # Load images as grayscale and normalize
        images = []
        for file in files:
            img = Image.open(file).convert("L")  # Convert to grayscale
            images.append(np.array(img, dtype=np.float32))

        # Stack images into a 3D array
        image_stack = np.stack(images, axis=0)
        image_stack = (image_stack - image_stack.min()) / (image_stack.max() - image_stack.min())

        # Apply temporal color coding
        global colored_images_stack, mip_array
        colored_images_stack = []
        for i, img in enumerate(image_stack):
            rgba_color = np.array(selected_colormap(i / (len(image_stack) - 1))[:3])  # Extract RGB
            color_coded = (img[:, :, None] * rgba_color * 255).astype(np.uint8)  # Apply colormap to grayscale
            color_coded_img = Image.fromarray(color_coded, mode="RGB")

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(color_coded_img)
            enhanced_img = enhancer.enhance(2.0)  # Adjust contrast factor
            colored_images_stack.append(enhanced_img)

        # Create a maximum intensity projection (MIP)
        mip_array = np.max(np.stack([np.array(img) for img in colored_images_stack], axis=0), axis=0)

        # Save MIP images
        mip_output_path = os.path.join(folder, "MIP_no_colorbar.png")
        Image.fromarray(mip_array).save(mip_output_path)

        mip_with_colorbar_output_path = os.path.join(folder, "MIP_with_colorbar.png")
        save_mip_with_colorbar(mip_with_colorbar_output_path, selected_colormap_name)

        # Save as a stack compatible with ImageJ
        stack_output_path = os.path.join(folder, "temporal_color_stack.tif")
        colored_images_stack[0].save(
            stack_output_path,
            save_all=True,
            append_images=colored_images_stack[1:],
            compression="tiff_deflate"
        )
        messagebox.showinfo("Success", f"Results saved to:\n{folder}")

        # Display MIP
        display_mip()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def save_mip_with_colorbar(output_path, colormap_name):
    """Saves the MIP with a color bar."""
    if mip_array is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mip_array, aspect='auto')
        ax.axis("off")

        # Add color bar
        norm = plt.Normalize(vmin=0, vmax=len(colored_images_stack) - 1)
        colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(colormap_name)), ax=ax, shrink=0.8)
        colorbar.set_label("Time", fontsize=10)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def display_mip():
    """Displays the MIP image in the MIP frame."""
    if mip_array is not None:
        mip_image = Image.fromarray(mip_array)
        mip_image_resized = mip_image.resize((300, 300), Image.NEAREST)  # Resize for display
        mip_tk = ImageTk.PhotoImage(mip_image_resized)

        mip_label.config(image=mip_tk)
        mip_label.image = mip_tk


video_playing = False


def start_video():
    """Starts playing the processed image stack as a video."""
    global video_playing
    if not colored_images_stack:
        messagebox.showerror("Error", "No processed image stack available to play.")
        return

    if not video_playing:
        video_playing = True
        play_video(0)


def stop_video():
    """Stops video playback."""
    global video_playing
    video_playing = False


def play_video(index):
    """Plays the video loop."""
    global video_playing
    if video_playing and colored_images_stack:
        img = colored_images_stack[index]
        img_resized = img.resize((300, 300), Image.NEAREST)  # Resize for display
        img_tk = ImageTk.PhotoImage(img_resized)
        video_label.config(image=img_tk)
        video_label.image = img_tk

        next_index = (index + 1) % len(colored_images_stack)
        root.after(100, play_video, next_index)  # 10 FPS

from PIL import ImageTk

def save_current_frame():
    """Saves the current frame being displayed in the video."""
    if video_label.image is None:
        messagebox.showerror("Error", "No frame to save.")
        return

    folder = output_folder.get()
    if not folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    # Extract the original PIL Image from PhotoImage
    img = video_label.image
    if isinstance(img, ImageTk.PhotoImage):
        pil_img = ImageTk.getimage(img)  # Convert PhotoImage to PIL Image

    # Save the image as PNG
    frame_output_path = os.path.join(folder, "current_video_frame.png")
    pil_img.save(frame_output_path)  # Save the PIL Image directly
    messagebox.showinfo("Saved", f"Current frame saved to:\n{frame_output_path}")



# Tkinter GUI
root = tk.Tk()
root.title("Temporal Color Coding Tool")
root.geometry("650x550")  # Adjust size for compact display

colored_images_stack = []  # Global variable to store processed images
mip_array = None  # Global variable for MIP
image_files = tk.StringVar()
output_folder = tk.StringVar()

# Main frames
top_frame = tk.Frame(root, padx=10, pady=10)
top_frame.pack(side=tk.TOP, fill=tk.X)

middle_frame = tk.Frame(root, padx=10, pady=10)
middle_frame.pack(side=tk.TOP, fill=tk.X)

bottom_frame = tk.Frame(root, padx=10, pady=10)
bottom_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
# Top frame widgets
tk.Label(top_frame, text="Selected Image Files:").grid(row=0, column=0, sticky="w")
selected_files_label = tk.Label(top_frame, text="No files selected.")
selected_files_label.grid(row=0, column=1, sticky="w")
tk.Button(top_frame, text="Select Files", command=select_images).grid(row=0, column=2, padx=5)

tk.Label(top_frame, text="Output Folder:").grid(row=1, column=0, sticky="w")
tk.Entry(top_frame, textvariable=output_folder, width=40).grid(row=1, column=1, sticky="w")
tk.Button(top_frame, text="Select Folder", command=select_output_folder).grid(row=1, column=2, padx=5)

# Colormap selection
tk.Label(top_frame, text="Colormap:").grid(row=2, column=0, sticky="w")
colormap_var = tk.StringVar(value="viridis")
colormap_menu = ttk.Combobox(top_frame, textvariable=colormap_var, values=plt.colormaps(), state="readonly", width=20)
colormap_menu.grid(row=2, column=1, sticky="w")
colormap_menu.bind("<<ComboboxSelected>>", lambda e: reset_gui())

# Apply button
apply_button = tk.Button(top_frame, text="Apply Temporal Color Coding", command=apply_temporal_color_coding)
apply_button.grid(row=3, column=0, columnspan=3, pady=10)

# MIP display
tk.Label(middle_frame, text="MIP:").grid(row=0, column=0)
mip_label = tk.Label(middle_frame, bg="black", width=300, height=300)
mip_label.grid(row=1, column=0, padx=5, pady=5)
clear_placeholder(mip_label)

# Video controls
tk.Label(middle_frame, text="Video:").grid(row=0, column=1)
video_label = tk.Label(middle_frame, bg="black", width=300, height=300)
video_label.grid(row=1, column=1, padx=5, pady=5)
clear_placeholder(video_label)

tk.Button(bottom_frame, text="Start Video", command=start_video).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(bottom_frame, text="Stop Video", command=stop_video).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(bottom_frame, text="Save Current Frame", command=save_current_frame).pack(side=tk.LEFT, padx=5, pady=5)

# Run the application
root.mainloop()
