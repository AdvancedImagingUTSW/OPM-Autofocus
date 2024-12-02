# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:33:33 2024

@author: S233755
"""

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from matplotlib.figure import Figure

global images

global x_coords
global y_coords
global max_coords
    
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
    global images
    global x_coords
    global y_coords
    global max_coords
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
        x_coords = []
        y_coords = []
        max_coords = []
        for file in files:
            img = Image.open(file).convert("L")  # Convert to grayscale
            img = np.array(img, dtype=np.float32)
            images.append(img)
            x0, y0, max0 = process_image(img)
            x_coords.append(x0)
            y_coords.append(y0)
            max_coords.append(max0)

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
        process_and_plot_psf_data(x_coords, y_coords,plot_frame)
        # display_colorbar(selected_colormap_name)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Define 1D Gaussian function for fitting
def gaussian_1d(x, mean, sigma, amplitude, offset):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset

# Function to perform 1D Gaussian fit on the x and y projections
def fit_1d_gaussian(image, roi_size=50):
    # Find the brightest pixel location
    brightest_pixel = np.unravel_index(np.argmax(image, axis=None), image.shape)
    y_bright, x_bright = brightest_pixel
    
    # Define ROI boundaries around the brightest pixel
    x_min = max(0, x_bright - roi_size // 2)
    x_max = min(image.shape[1], x_bright + roi_size // 2)
    y_min = max(0, y_bright - roi_size // 2)
    y_max = min(image.shape[0], y_bright + roi_size // 2)
    
    # Extract the ROI
    roi = image[y_min:y_max, x_min:x_max]
    
    # Sum projections along x and y axes
    x_projection = np.sum(roi, axis=0)
    y_projection = np.sum(roi, axis=1)
    
    # # Sum projections along x and y axes
    # x_projection = np.mean(roi, axis=0)
    # y_projection = np.mean(roi, axis=1)
    
    
    # Initial guesses for Gaussian fitting
    x_initial_guess = (roi.shape[1] // 2, 10, np.max(x_projection), np.min(x_projection))
    y_initial_guess = (roi.shape[0] // 2, 10, np.max(y_projection), np.min(y_projection))
    
    # Fit Gaussian to x and y projections
    try:
        popt_x, _ = curve_fit(gaussian_1d, np.arange(len(x_projection)), x_projection, p0=x_initial_guess)
        popt_y, _ = curve_fit(gaussian_1d, np.arange(len(y_projection)), y_projection, p0=y_initial_guess)
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None, None

    # Calculate the center in the original image coordinates
    x_center = x_min + popt_x[0]
    y_center = y_min + popt_y[0]
    
    return x_center, y_center

# Function to process a single image and plot result
def process_image(image):
    # Load the image in grayscale (16-bit)
    max0 = np.max(image)
    
    # Ensure the image is loaded and non-zero pixels exist
    if image is None or np.count_nonzero(image) == 0:
        print("Failed to load or empty image")
        return None, None

    # Normalize the image for better fitting
    image_normalized = image.astype(float) / image.max()    
   
    # Perform 1D Gaussian fitting
    x0, y0 = fit_1d_gaussian(image_normalized)
        
    return x0, y0, max0

def process_and_plot_psf_data(com_x_coords, com_y_coords, frame, window_size=1, range_start = 100, range_stop = 400):
    """
    Processes and plots PSF data, saves the results to a CSV file, and computes error metrics.

    Parameters:
    - x_index (array): Array of z-position indices.
    - com_x_coords (array): Array of x-coordinate values.
    - com_y_coords (array): Array of y-coordinate values.
    - output_csv (str): Filename for saving the output CSV file.
    - window_size (int): Window size for moving average smoothing.

    Returns:
    - index_ (array): Processed z-position values.
    - x_coords_ (array): Smoothed x-coordinate values.
    - y_coords_ (array): Smoothed y-coordinate values.
    - error_corrected (array): Corrected error values.
    """
    # Helper function for moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    # Calculate the middle index dynamically
    middle_index = len(com_x_coords) // 2
    x_index = np.arange(len(com_x_coords))
    # Process data
    index_ = x_index * (5 / (len(x_index) - 1)) - 2.5
    
    x_coords_ = moving_average(com_x_coords - com_x_coords[middle_index], window_size)
    y_coords_ = moving_average(com_y_coords - com_y_coords[middle_index], window_size)

    
    # Compute error
    error = np.linalg.norm(
        np.stack([x_coords_, y_coords_], axis=1) - np.array([x_coords_[middle_index], y_coords_[middle_index]]),
        axis=1
    )
    error_corrected = error.copy()
    error_corrected[middle_index:] = -error_corrected[middle_index:]


    # Clear any existing widgets in the frame
    for widget in frame.winfo_children():
        widget.destroy()
    
    # Create a Matplotlib figure
    #fig = Figure(figsize=(3, 3))
    
    # # Add first subplot for XY data
    # ax1 = fig.add_subplot(121)  # 1 row, 2 columns, 1st plot
    # ax1.plot(index_[range_start:range_stop], x_coords_[range_start:range_stop], color='black', label='X displacement', linewidth=0.2, linestyle='--')
    # ax1.plot(index_[range_start:range_stop], y_coords_[range_start:range_stop], color='red', label='Y displacement', linewidth=0.2, linestyle='--')
    # ax1.set_title('x - y displacement of PSF on PiCam corresponding to z-position')
    # ax1.set_xlabel('z-position (microns)')
    # ax1.set_ylabel('Displacement from setpoint (pixel)')
    # ax1.grid(True)
    
    # # Add second subplot for Error data
    # ax2 = fig.add_subplot(122)  # 1 row, 2 columns, 2nd plot
    # ax2.plot(index_[range_start:range_stop], error_corrected[range_start:range_stop], color='black', label='Euclidean distance - calibration curves', linewidth=0.2, linestyle='--')
    # ax2.set_title('Error')
    # ax2.set_xlabel('z-position (microns)')
    # ax2.set_ylabel('Euclidean distance from setpoint (pixel)')
    # ax2.grid(True)
    
    # Create a Matplotlib figure
    fig = Figure(figsize=(6, 3))  # Wider for two subplots side by side
    
    # Adjust the layout for a tighter fit
    fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.95, top=0.9, bottom=0.2)
    
    # Add first subplot for XY data
    ax1 = fig.add_subplot(121)  # 1 row, 2 columns, 1st plot
    ax1.plot(index_[range_start:range_stop], x_coords_[range_start:range_stop], color='black', label='X displacement', linewidth=0.5, linestyle='--')
    ax1.plot(index_[range_start:range_stop], y_coords_[range_start:range_stop], color='red', label='Y displacement', linewidth=0.5, linestyle='--')
    ax1.set_title('XY Displacement of PSF', fontsize=10)
    ax1.set_xlabel('Z-position (microns)', fontsize=8)
    ax1.set_ylabel('Displacement (pixels)', fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax1.grid(True, linewidth=0.5, linestyle='--')
    ax1.legend(fontsize=6)
    
    # Add second subplot for Error data
    ax2 = fig.add_subplot(122)  # 1 row, 2 columns, 2nd plot
    ax2.plot(index_[range_start:range_stop], error_corrected[range_start:range_stop], color='blue', label='Euclidean distance', linewidth=0.5, linestyle='--')
    ax2.set_title('Calibration curve', fontsize=10)
    ax2.set_xlabel('Z-position (microns)', fontsize=8)
    ax2.set_ylabel('Euclidian distance (pixels)', fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax2.grid(True, linewidth=0.5, linestyle='--')
    ax2.legend(fontsize=6)
    
    # Embed the Matplotlib figure in the GUI
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    
    
    return index_, x_coords_, y_coords_, error_corrected

        
# def display_colorbar(colormap_name):
#     """Displays the color bar next to the MIP display."""
#     fig, ax = plt.subplots(figsize=(2, 5))
#     norm = plt.Normalize(vmin=0, vmax=len(colored_images_stack) - 1)
#     colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(colormap_name)), ax=ax)
#     colorbar.set_label("Time", fontsize=10)

#     # Convert Matplotlib figure to PIL Image
#     fig.canvas.draw()
#     width, height = fig.canvas.get_width_height()
#     img = Image.frombytes('RGB', (width, height), fig.canvas.tostring_rgb())

#     # Display in the colorbar_label
#     img_tk = ImageTk.PhotoImage(img.resize((50, 300), Image.NEAREST))
#     colorbar_label.config(image=img_tk)
#     colorbar_label.image = img_tk

#     plt.close(fig)


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
root.title("OPM-Autofocus Analysis and Visualize")
root.geometry("645x900")  # Adjust size for compact display

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

plot_frame = tk.Frame(root, padx=10, pady=10)
plot_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)  # New frame for plots
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

# Middle frame widgets
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

# Bottom frame widget
tk.Button(bottom_frame, text="Start Video", command=start_video).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(bottom_frame, text="Stop Video", command=stop_video).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(bottom_frame, text="Save Current Frame", command=save_current_frame).pack(side=tk.LEFT, padx=5, pady=5)
# Add "Update Plot" button for testing
# tk.Button(bottom_frame, text="Update Plot", command=lambda: update_plot(
#     np.arange(500),  # Example x_index
#     np.sin(np.linspace(0, 10, 500)) * 10 + 5,  # Example com_x_coords
#     np.cos(np.linspace(0, 10, 500)) * 10 + 5   # Example com_y_coords
# )).pack(side=tk.LEFT, padx=5, pady=5)

# Plot frame widget

# x - y displacement

# tk.Label(plot_frame, text="Euclidian Error:").grid(row=0, column=0)
# xy_label = tk.Label(plot_frame, bg="black", width=300, height=300)
# xy_label.grid(row=1, column=0, padx=5, pady=5)
# clear_placeholder(xy_label)

# # Error function
# tk.Label(plot_frame, text="Euclidian Error:").grid(row=0, column=1)
# error_label = tk.Label(plot_frame, bg="black", width=300, height=300)
# error_label.grid(row=1, column=1, padx=5, pady=5)
# clear_placeholder(error_label)

# tk.Label(middle_frame, text="X' - Y' displacement:").grid(row=0, column=2)
# xy_label = tk.Label(middle_frame, bg="black", width=300, height=300)
# xy_label.grid(row=1, column=2, padx=5, pady=5)
# clear_placeholder(xy_label)

# # Error function
# tk.Label(middle_frame, text="Euclidian Distance - Error:").grid(row=0, column=3)
# error_label = tk.Label(middle_frame, bg="black", width=300, height=300)
# error_label.grid(row=1, column=3, padx=5, pady=5)
# clear_placeholder(error_label)

# plot_frame = ttk.Frame(root, padding="10")
# plot_frame.pack(fill=tk.BOTH, expand=True)


# Run the application
root.mainloop()
