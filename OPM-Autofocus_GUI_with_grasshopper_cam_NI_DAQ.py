import os

import time

import numpy as np

import cv2

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from tkinter import messagebox

import csv

import nidaqmx

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import PySpin
from threading import Thread

from PIL import Image, ImageTk
import tkinter as tk

import numpy as np
from scipy.optimize import curve_fit
import glob

global sigma 
sigma = 50
global wait_time 
wait_time = 0.02
RT = 8000	# How long to run in seconds
global is_calib
global pos
global prop
prop = 10
global integral_gain
integral_gain = 1
global derivative_gain
derivative_gain = 0
global frame_count
global error
global integral
global control_signal
global previous_error
global im, tt, t0, t100
control_signal =0
integral = 0
previous_error = 0

global is_print
is_print = 0

frame_count = 0
t100 = time.time()

tt = time.time()
t0 = tt


global pid
global logger

global xy_profile
xy_profile = np.zeros(shape=(3, 2))
global distance_left_to_center
distance_left_to_center = 0
global distance_right_to_center
distance_right_to_center = 0

global picam_running
picam_running = 0
global set_preview
set_preview = 1
global is_calib
is_calib = 0

# -----------------------------------------------------------------------------
scanrange =  25	#In micron
step_size = 100	#In nanometers
V_cal = 0.01	#nm per voltage (piezo specific)
VDD = 2.5			#Check your DAC voltage at max value 65536 ours is 2.5
V0 = int(65536/2)	#Starting value for DAC = 1.25V


pos = V0


ao_task = nidaqmx.Task()  # Use explicit initialization without 'with'
channel_name = "Dev1/ao0"  # Update with your device name
# Add an analog output voltage channel
ao_task.ao_channels.add_ao_voltage_chan(channel_name)
ao_task.write(1.25)   
ao_task.stop()  # Stop the task
ao_task.close()  # Release resource


size = (640, 400)
# size = (320, 200)
flim = 500
exp = 10
gain = 0.5



# Define 1D Gaussian function for fitting
def gaussian_1d(x, mean, sigma, amplitude, offset):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset

# Function to perform 1D Gaussian fit on the x and y projections
def calculate_psf_center(image, roi_size=100):
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

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.previous_error = 0
        self.integral = 0

    def update(self, setpoint, cx_current, cy_current, dt):
        # Calculate the error
        if cx_current < xy_profile[0, 0]:
                error = -np.linalg.norm([cx_current, cy_current] - setpoint)
        else:
                error = np.linalg.norm([cx_current, cy_current] - setpoint)
        
        # print(setpoint)
        # print(current_psf_position)

        # Proportional term
        p_term = self.kp * error

        # Integral term (sum of error over time)
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term (rate of change of error)
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Calculate total control signal
        control_signal = p_term + i_term + d_term

        # Save error for next derivative calculation
        self.previous_error = error

        return control_signal, self.previous_error, self.integral
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
            

# Function to zoom in on the center of the image
def zoom_in_image(image, zoom_ratio = 3):
    height, width = image.shape  # Get the dimensions of the image
    # print(height)
    # print(width)
    # Calculate the size of the zoomed-in region
    zoomed_width = int(width / zoom_ratio)
    zoomed_height = int(height / zoom_ratio)

    # Calculate the coordinates of the center of the image
    center_x, center_y = width // 2, height // 2

    # Define the top-left and bottom-right points of the zoomed-in section
    x1 = max(center_x - zoomed_width // 2, 0)
    y1 = max(center_y - zoomed_height // 2, 0)
    x2 = min(center_x + zoomed_width // 2, width)
    y2 = min(center_y + zoomed_height // 2, height)

    # Crop the zoomed-in region
    zoomed_image = image[y1:y2, x1:x2]

    # Resize the cropped image back to the original size (or desired display size)
    zoomed_image_resized = cv2.resize(zoomed_image, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_image_resized

class AutoFocusLogger:
    def __init__(self):
        self.logs = []  # Store logs in memory (RAM)
        self.index = 0  # Initialize log index

    def log(self, cx_current, cy_current, control_signal, pos):
        # Record a log entry in memory with index
        self.logs.append([self.index, cx_current, cy_current, control_signal, pos])
        self.index += 1

    def save_logs(self, log_filename="autofocus_log.csv"):
        # Save all logs to a CSV file
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "cx_current", "cy_current", "control_signal", "pos"])  # Write header
            writer.writerows(self.logs)  # Write all logs from memory
        print(f"Logs saved to {log_filename}")
            
def dump_callback(request):
    pass           

def calib_psf(image):
    global xy_profile, distance_left_to_center, distance_right_to_center
    #dac.value = V0
    time.sleep(3)	# Give time to adjust focus
    cx,cy= calculate_psf_center(image)
    xy_profile[0] = [cx,cy] 
    print(cx, cy)

# Tkinter GUI setup
class Application(tk.Tk):
    def __init__(self):
        global logger, is_calib
        super().__init__()
        self.title("Autofocus GUI")
        self.running = False
        self.camera_initialized = False
        self.image_event_handler = None
        
        #self.geometry("840x650")
        self.zoom_ratio = 4  # Default zoom ratio
        #self.attributes('-fullscreen', True)
        self.previous_error = 0
        self.integral = 0
        self.create_widgets()
        self.update_preview()
        
        logger = AutoFocusLogger()
        self.is_running = False
        
        # FLIR Camera Initialization
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            print("No camera detected.")
            self.cam_list.Clear()
            self.system.ReleaseInstance()
        else:
            self.cam = self.cam_list[0]
            self.cam.Init()
            self.camera_initialized = True
            if self.cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                messagebox.showerror("Error", "Unable to enable automatic exposure.")
                return

            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
            #cam.DeInit()
            self.set_format()
            
            # self.set_binning(4)
            # offset_x = 1728  # Calculated center offset X
            # offset_y = 1260  # Calculated center offset Y
            # width = 640      # Desired ROI width
            # height = 480     # Desired ROI height
            # self.configure_roi(self.cam, offset_x, offset_y, width, height)
            # messagebox.showinfo("Success", "Exposure reset to automatic.")
            
            print("Camera initialized.")


    def create_widgets(self):
        

        # Create and place the image preview label
        self.canvas = tk.Canvas(self, width=640, height=480, bg="black")
        self.canvas.grid(row=0, column=0, rowspan=8, padx=5, pady=5)  # Image preview

        # Create control buttons
        self.preview_button = tk.Button(self, text="Preview", command=self.preview)
        self.preview_button.grid(row=1, column=1, padx=5, pady=5)

        self.reset_piezo = tk.Button(self, text="Reset", command=self.reset)
        self.reset_piezo.grid(row=2, column=1, padx=5, pady=5)

        self.calib_button = tk.Button(self, text="Calib", command=self.calib)
        self.calib_button.grid(row=3, column=1, padx=5, pady=5)

        self.start_button = tk.Button(self, text="Start", command=self.start_recording)
        self.start_button.grid(row=4, column=1, padx=5, pady=5)

        self.stop_button = tk.Button(self, text="Stop", command=self.stop_recording)
        self.stop_button.grid(row=5, column=1, padx=5, pady=5)

        self.set_button_x4 = tk.Button(self, text="X4", command=self.set_zoom_4)
        self.set_button_x4.grid(row=6, column=2, padx=5, pady=5)
        self.set_button_x1 = tk.Button(self, text="X1", command=self.set_zoom_1)
        self.set_button_x1.grid(row=7, column=1, padx=5, pady=5)
        self.set_button_x10 = tk.Button(self, text="X10", command=self.set_zoom_10)
        self.set_button_x10.grid(row=7, column=2, padx=5, pady=5)
        
        self.exposure_label = tk.Label(self, text="Camera Exposure")
        self.exposure_label.grid(row=0, column=3, padx=5, pady=5)
        
        self.exposure_entry = tk.Entry(self)
        self.exposure_entry.grid(row=1, column=3, padx=5, pady=5)
        self.exposure_entry.insert(0, str(20000))

        self.set_exposure_button = tk.Button(self, text="Set Exposure", command=self.set_exposure)
        self.set_exposure_button.grid(row=2, column=3, padx=5, pady=5)
        
        self.reset_button = tk.Button(self, text="Reset Exposure", command=self.reset_exposure)
        self.reset_button.grid(row=3, column=3, padx=5, pady=5)
        
        self.binning_label = tk.Label(self, text="Camera Binning")
        self.binning_label.grid(row=4, column=3, padx=5, pady=5)
        
        self.binning_entry = tk.Entry(self)
        self.binning_entry.grid(row=5, column=3, padx=5, pady=5)
        self.binning_entry.insert(0, str(2))

        self.set_binning_button = tk.Button(self, text="Set binning", command=self.set_binning)
        self.set_binning_button.grid(row=6, column=3, padx=5, pady=5)
        
        # Create and place text entries for PID parameters in a new column
        self.prop_label = tk.Label(self, text="Kp")
        self.prop_label.grid(row=0, column=2, sticky=tk.W)
        self.prop_entry = tk.Entry(self, width=5)  # Set width to 5 characters
        self.prop_entry.grid(row=1, column=2)
        self.prop_entry.insert(0, str(prop))  # Default value

        self.integral_gain_label = tk.Label(self, text="Ki")
        self.integral_gain_label.grid(row=2, column=2, sticky=tk.W)
        self.integral_gain_entry = tk.Entry(self, width=5)  # Set width to 5 characters
        self.integral_gain_entry.grid(row=3, column=2)
        self.integral_gain_entry.insert(0, str(integral_gain))  # Default value

        self.derivative_gain_label = tk.Label(self, text="Kd")
        self.derivative_gain_label.grid(row=4, column=2, sticky=tk.W)
        self.derivative_gain_entry = tk.Entry(self, width=5)  # Set width to 5 characters
        self.derivative_gain_entry.grid(row=5, column=2)
        self.derivative_gain_entry.insert(0, str(derivative_gain))  # Default value

        self.zoom_label = tk.Label(self, text="Zoom:")
        self.zoom_label.grid(row=6, column=1, sticky=tk.W)
        
    def set_exposure(self):
        try:
            #cam = self.get_camera()
            exposure_time = float(self.exposure_entry.get())

            #cam.Init()

            if self.cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                messagebox.showerror("Error", "Unable to disable automatic exposure.")
                return

            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

            if self.cam.ExposureTime.GetAccessMode() != PySpin.RW:
                messagebox.showerror("Error", "Unable to set exposure time.")
                return

            exposure_time = min(self.cam.ExposureTime.GetMax(), exposure_time)
            self.cam.ExposureTime.SetValue(exposure_time)
            #cam.DeInit()

            messagebox.showinfo("Success", f"Exposure time set to {exposure_time} microseconds.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def set_format(self):
        try:
            if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
                print('Pixel format set to %s...' % self.cam.PixelFormat.GetCurrentEntry().GetSymbolic())
            else:
                print('Pixel format not available...')
        except Exception as e:
            messagebox.showerror("Error", str(e))
     
    def configure_roi(self,camera, offset_x, offset_y, width, height):
        """
        Configures the ROI for the camera.
        :param camera: PySpin Camera object.
        :param offset_x: Horizontal offset of the ROI.
        :param offset_y: Vertical offset of the ROI.
        :param width: Width of the ROI.
        :param height: Height of the ROI.
        """
        try:
            # Get the node map of the camera
            nodemap = camera.GetNodeMap()

            # Set ROI Width
            width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
                width_node.SetValue(width)
                print(f"Width set to {width}")

            # Set ROI Height
            height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
                height_node.SetValue(height)
                print(f"Height set to {height}")


            # Set Offset X
            offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
            if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
                offset_x_node.SetValue(offset_x)
                print(f"OffsetX set to {offset_x}")

            # Set Offset Y
            offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
            if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
                offset_y_node.SetValue(offset_y)
                print(f"OffsetY set to {offset_y}")

            
        except PySpin.SpinnakerException as ex:
            print(f"Error: {ex}")
        
     
    def set_binning(self):
        self.cam.EndAcquisition()
        binning_ratio = int(self.binning_entry.get())
        nodemap = self.cam.GetNodeMap()
        
        binning_horizontal = PySpin.CIntegerPtr(nodemap.GetNode("BinningHorizontal"))
        binning_vertical = PySpin.CIntegerPtr(nodemap.GetNode("BinningVertical"))

        if PySpin.IsAvailable(binning_horizontal) and PySpin.IsWritable(binning_horizontal):
            binning_horizontal.SetValue(binning_ratio)  # Set horizontal binning to 2

        if PySpin.IsAvailable(binning_vertical) and PySpin.IsWritable(binning_vertical):
            binning_vertical.SetValue(binning_ratio)  # Set vertical binning to 2

        # binning_horizontal = PySpin.CIntegerPtr(nodemap.GetNode("BinningHorizontal"))
        # binning_vertical = PySpin.CIntegerPtr(nodemap.GetNode("BinningVertical"))

        # if PySpin.IsAvailable(binning_horizontal) and PySpin.IsWritable(binning_horizontal):
        #     binning_horizontal.SetValue(2)  # Set horizontal binning to 2

        # if PySpin.IsAvailable(binning_vertical) and PySpin.IsWritable(binning_vertical):
        #     binning_vertical.SetValue(2)  # Set vertical binning to 2

        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        continuous_mode = PySpin.CEnumEntryPtr(acquisition_mode.GetEntryByName("Continuous"))

        if PySpin.IsAvailable(continuous_mode) and PySpin.IsReadable(continuous_mode):
            acquisition_mode.SetIntValue(continuous_mode.GetValue())
        self.cam.BeginAcquisition()
    def reset_exposure(self):
        try:
            #cam = self.get_camera()
            #cam.Init()
    
            if self.cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                messagebox.showerror("Error", "Unable to enable automatic exposure.")
                return
    
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
            #cam.DeInit()
    
            messagebox.showinfo("Success", "Exposure reset to automatic.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_preview(self):
        print()
        # global set_preview
        # if set_preview == 1:
            # # Capture an image frame from Picamera2
            # raw = picam2.capture_array("raw")
            # image = np.copy(raw.view('uint16'))
            
            # # Apply the zoom function to zoom into the center
            # image = zoom_in_image(image, self.zoom_ratio)
            
            # # Convert to a format Tkinter can display
            # image = Image.fromarray(image)
            # #image = image.resize((320, 240), Image.ANTIALIAS)
            # photo = ImageTk.PhotoImage(image)
            
            # # Update the image in the label
            # self.image_label.configure(image=photo)
            # self.image_label.image = photo
            # # Schedule the next update            
            # self.after(100, self.update_preview)
            
    
    def calib(self):
        global is_calib
        
        ao_task = nidaqmx.Task()  # Use explicit initialization without 'with'
        channel_name = "Dev1/ao0"  # Update with your device name
        # Add an analog output voltage channel
        ao_task.ao_channels.add_ao_voltage_chan(channel_name)
        ao_task.write(1.25)   
        ao_task.stop()  # Stop the task
        ao_task.close()  # Release resource
        
        image = self.image_event_handler.pass_image
        
        calib_psf(image)
        global set_preview, picam_running, prop, integral_gain,derivative_gain, pid, logger
        # self.is_running = True
        # print("Autofocus started...")
        # Retrieve values from text entries
        try:
            prop = float(self.prop_entry.get())
            integral_gain = float(self.integral_gain_entry.get())
            derivative_gain = float(self.derivative_gain_entry.get())
        except ValueError:
            print("Invalid input. Using default values.")
        
        pid = PIDController(kp=prop, ki=integral_gain, kd=derivative_gain)  # Set your own PID values
        is_calib = True    
        print("Calibrated!")
        
    def start_recording(self):
        print()
        if not self.camera_initialized:
            print("Camera not initialized.")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        
        
        # Register the image event handler
        self.image_event_handler = ImageEventHandler(self.cam)
        self.cam.RegisterEventHandler(self.image_event_handler)
        self.cam.BeginAcquisition()
        print("Acquisition started.")
        self.update_canvas()
        # time.sleep(wait_time)
        # if self.image_event_handler.last_image:
            
        # while self.wait_image:
        #     if self.image_event_handler.last_image:
        #         self.update_canvas(self.image_event_handler.last_image, self.image_event_handler.pass_image)
        #         self.wait_image = False
        
        #self.after(10000,self.update_canvas(self.image_event_handler.last_image, self.image_event_handler.pass_image))

        # except PySpin.SpinnakerException as ex:
        #     print(f"Error during acquisition: {ex}")

            
    def stop_recording(self):
        print()
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        
        self.cam.EndAcquisition()
        self.cam.UnregisterEventHandler(self.image_event_handler)

        self.stop_button.config(state=tk.DISABLED)
        
        
    def preview(self):
        print()
        # global set_preview, picam_running
        # self.after_cancel(self.update_preview)
        # if picam_running == 0:
        #     picam2.start()
        # set_preview = 1
        # self.update_preview()
    
    def reset(self):
        
        global pid
        ao_task = nidaqmx.Task()  # Use explicit initialization without 'with'
        channel_name = "Dev1/ao0"  # Update with your device name
        # Add an analog output voltage channel
        ao_task.ao_channels.add_ao_voltage_chan(channel_name)
        ao_task.write(1.25)   
        ao_task.stop()  # Stop the task
        ao_task.close()  # Release resource
        pid.reset()
        
    def set_zoom_4(self):
        self.zoom_ratio = 4
    
    def set_zoom_1(self):
        self.zoom_ratio = 1
    
    def set_zoom_10(self):
        self.zoom_ratio = 10  
        
    # def acquire_images(self):
    #     """
    #     Acquires images using the FLIR camera and updates the canvas with the latest images.
    #     """
    #     try:
    #         # Register the image event handler
    #         self.image_event_handler = ImageEventHandler(self.cam)
    #         self.cam.RegisterEventHandler(self.image_event_handler)
    #         self.cam.BeginAcquisition()
    #         print("Acquisition started.")

    #         while self.running:
    #             if self.image_event_handler.last_image:
    #                 self.update_canvas(self.image_event_handler.last_image, self.image_event_handler.pass_image)


    #         self.cam.EndAcquisition()
    #         self.cam.UnregisterEventHandler(self.image_event_handler)

    #     except PySpin.SpinnakerException as ex:
    #         print(f"Error during acquisition: {ex}")
    

    def update_canvas(self):
        if self.image_event_handler.last_image:
            pass_image = zoom_in_image(self.image_event_handler.pass_image,self.zoom_ratio)
             # Convert to PIL image
            pass_image = pass_image/65535*255
             #print(pass_image)
            pass_image= Image.fromarray(pass_image).resize((640, 480))
            tk_image = ImageTk.PhotoImage(pass_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image
        self.after(100, self.update_canvas)

    def __del__(self):
        """
        Cleanup resources.
        """
        if self.camera_initialized:
            self.cam.DeInit()
            self.cam_list.Clear()
            self.system.ReleaseInstance()
        
class ImageEventHandler(PySpin.ImageEventHandler):
    """
    Handles image events from the FLIR camera and processes them for display.
    """

    def __init__(self, cam):
        super().__init__()
        self.last_image = None
        self.pass_image = None
        self._processor = PySpin.ImageProcessor()
        self._processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    def OnImageEvent(self, image):
        """
        This method defines an image event.
        """
        global is_calib, frame_count, im, pos, tt, t0, t100, prop, wait_time, integral, previous_error, logger
        if image.IsIncomplete():
            print("Image incomplete with image status %i..." % image.GetImageStatus())
        else:
            # Convert to Mono8
            
            image_array = image.GetNDArray()
            self.pass_image = image_array
            # Convert to PIL image
            #print(image_array)
            self.last_image = Image.fromarray(image_array).resize((640, 480))
        
            if is_calib == True:
                
                
                tt = time.time()
                frame_count = frame_count + 1
                
                
                cx_current, cy_current = calculate_psf_center(image_array)
                #print(cx_current)
                #print(cy_current)
                # # Example values
    
                dt = 0.02  # Time step
                if cx_current == None:
                    print("Can't fit PSF")
                    control_signal=None
                    new_dac_value =None
                else:
                    # Calculate control signal
                    control_signal,_error, _integral = pid.update(xy_profile[0], cx_current, cy_current, dt)
                    
                    # print("Control_signal: ", str(control_signal))
                    # print("Error: ", str(_error))
                    # print("Integral: ", str(_integral))
                    # print("Max intensit: ", str(np.max(image_array)))
                
                    pos -= control_signal
                    #print("Pos-signal: ",pos)
                    
                    if int(pos) < 1000:
                            #print("Positive")
                            #print(int(pos + control_signal))
                            print('out of compensation range')
                            pos = 1000
                    
                    if int(pos) > 64536:
                            #print("Negative")
                            #print(int(pos + control_signal))
                            print('out of compensation range')
                            pos = 64536
                            
                    new_dac_value = int(pos)
                    new_volt = float(new_dac_value/64536*2.5)
                    #dac.value = new_dac_value       
                    # Create an analog output task
                    ao_task = nidaqmx.Task()  # Use explicit initialization without 'with'
                    channel_name = "Dev1/ao0"  # Update with your device name
                    # Add an analog output voltage channel
                    ao_task.ao_channels.add_ao_voltage_chan(channel_name)
                    ao_task.write(new_volt)
                    ao_task.stop()  # Stop the task
                    ao_task.close()  # Release resource
                    
            #logger.log(cx_current, cy_current, control_signal, new_dac_value)
            #logger.log(cx_current, cy_current, control_signal, 0)
            
                    if frame_count % 30 == 0:
                        print("Control_signal: ", str(control_signal))
                        print("Error: ", str(_error))
                        print("Integral: ", str(_integral))
                        print("Max intensit: ", str(np.max(image_array)))
                    
                    #time.sleep(wait_time)
                    
                    if frame_count % 100 == 0:
                        print("FPS: " + str(100 / (time.time() - t100)))
                        t100 = time.time()
            
        

if __name__ == "__main__":
    app = Application()
    app.eval('tk::PlaceWindow . center')
    app.mainloop()



