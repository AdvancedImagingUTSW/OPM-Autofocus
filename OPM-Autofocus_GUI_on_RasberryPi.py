# It's important to select the correct prop and wait_time, prop under 0.3 should work without overshoot
# wait_time depend on the prop, if prop is higher, wait_time is higher (wait for dac.value update and stage to move) 
# the feed back work better with well define psf -> reduce the aperture down to 8 mm (full is 12 mm)
# reduce the aperture also increase the range of autofocusing (the good range has linear movement of PSF's centers)
# could consider include integral and derivative components
import busio
import board
import adafruit_ad569x

import os

import time

import numpy as np

import cv2

from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import Encoder
from picamera2.outputs import CircularOutput




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

global pos
global prop
prop = 0.1
global integral_gain
integral_gain = 0.01
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

# -----------------------------------------------------------------------------
scanrange =  25	#In micron
step_size = 100	#In nanometers
V_cal = 0.01	#nm per voltage (piezo specific)
VDD = 2.5			#Check your DAC voltage at max value 65536 ours is 2.5
V0 = int(65536/2)	#Starting value for DAC

# DAC settings
i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
dac = adafruit_ad569x.Adafruit_AD569x(i2c)
print("DAC settings are fine!")

pos = V0
dac.value = pos

picam2 = Picamera2()



size = (640, 400)
# size = (320, 200)
flim = 500
exp = 10
gain = 0.5


picam2.encode_stream_name = "raw"
video_config = picam2.create_video_configuration(raw={'format': 'R10', 'size': size}, controls = {"FrameDurationLimits": (500, 500), "ExposureTime": 60, "AnalogueGain": 1})
picam2.configure(video_config)
encoder = Encoder()
picam2.start()



# Define 1D Gaussian function for fitting
def gaussian_1d(x, mean, sigma, amplitude, offset):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset

# Function to perform 1D Gaussian fit on the x and y projections
def calculate_psf_center(image, roi_size=50):
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
    height, width = image.shape[:2]  # Get the dimensions of the image

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

import csv

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



def apply_timestamp(request):
    global frame_count, im, pos, tt, t0, t100, prop, wait_time, integral, previous_error, logger

    with MappedArray(request, "raw") as m:
        tt = time.time()
        frame_count = frame_count + 1
        im = np.copy(m.array.view('uint16'))  # get the whole image

        cx_current, cy_current = calculate_psf_center(im)
        
        
        # Example values

        dt = 0.02  # Time step
        if cx_current == None:
            print("Can't fit PSF")
            control_signal=None
            new_dac_value =None
        else:
            # Calculate control signal
            control_signal,_error, _integral = pid.update(xy_profile[0], cx_current, cy_current, dt)
            
            
            
            pos += control_signal*100
            
            
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
            dac.value = new_dac_value
        
        logger.log(cx_current, cy_current, control_signal, new_dac_value)
        #logger.log(cx_current, cy_current, control_signal, 0)
        
            
        if frame_count % 30 == 0:
            print("Control_signal: ", str(control_signal))
            print("Error: ", str(_error))
            print("Integral: ", str(_integral))
            print("Max intensit: ", str(np.max(im)))
        time.sleep(wait_time)
        
        if frame_count % 100 == 0:
            print("FPS: " + str(100 / (time.time() - t100)))
            t100 = time.time()
            
def dump_callback(request):
    pass           

def calib_psf():
    global xy_profile, distance_left_to_center, distance_right_to_center
    dac.value = V0
    time.sleep(3)	# Give time to adjust focus
    raw = picam2.capture_array("raw")	# Capture image array[50:350,320:920]
    arr = np.copy(raw.view('uint16'))	# 300x300 pixels centred on camera (2nd axis is double for bit reasons)
    cx,cy= calculate_psf_center(arr)
    xy_profile[0] = [cx,cy] 
    print(cx, cy)



global picam_running
picam_running = 0
global set_preview
set_preview = 1
global is_calib
is_calib = 0
# Tkinter GUI setup
class Application(tk.Tk):
    def __init__(self):
        global logger
        super().__init__()
        self.title("Autofocus GUI")
        #self.geometry("840x650")
        self.zoom_ratio = 4  # Default zoom ratio
        #self.attributes('-fullscreen', True)
        self.previous_error = 0
        self.integral = 0
        self.create_widgets()
        self.update_preview()
        
        logger = AutoFocusLogger()
        self.is_running = False


    def create_widgets(self):
        
        # Create and place the image preview label
        self.image_label = tk.Label(self)
        self.image_label.grid(row=0, column=0, rowspan=8, padx=5, pady=5)  # Image preview

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
        # self.zoom_entry = tk.Entry(self, width=5)  # Set width to 5 characters
        # self.zoom_entry.grid(row=7, column=2)
        # self.zoom_entry.insert(0, str(1))  # Default value



    def update_preview(self):
        global set_preview
        if set_preview == 1:
            # Capture an image frame from Picamera2
            raw = picam2.capture_array("raw")
            image = np.copy(raw.view('uint16'))
            
            # Apply the zoom function to zoom into the center
            image = zoom_in_image(image, self.zoom_ratio)
            
            # Convert to a format Tkinter can display
            image = Image.fromarray(image)
            #image = image.resize((320, 240), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            
            # Update the image in the label
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            # Schedule the next update            
            self.after(100, self.update_preview)
            
    
    def calib(self):
        global set_preview

        picam2.start()
#         set_preview = 1
#         self.update_preview()
        calib_psf()
        print("Calibrated!")
        
    def start_recording(self):
        global set_preview, picam_running, prop, integral_gain,derivative_gain, pid, logger
        self.is_running = True
        print("Autofocus started...")
        # Retrieve values from text entries
        try:
            prop = float(self.prop_entry.get())
            integral_gain = float(self.integral_gain_entry.get())
            derivative_gain = float(self.derivative_gain_entry.get())
        except ValueError:
            print("Invalid input. Using default values.")
        
        pid = PIDController(kp=prop, ki=integral_gain, kd=derivative_gain)  # Set your own PID values
            
        picam2.start_recording(encoder,CircularOutput())	# Start running
        picam2.pre_callback = apply_timestamp
        picam_running = 1
        #self.after_cancel(self.update_preview)
        set_preview = 1
        self.update_preview()

    def stop_recording(self):
        global set_preview, picam_running,logger
        self.is_running = False
        print("Autofocus stopped.")
        # Save logs to file
        logger.save_logs()
        #picam2.stop_recording() # End recording
        set_preview = 0
        self.after_cancel(self.update_preview)
        print("Stop!")
        picam2.pre_callback = dump_callback
        picam2.stop_recording()
        picam_running = 0
        #set_preview = 1
        
    def preview(self):
        global set_preview, picam_running
        self.after_cancel(self.update_preview)
        if picam_running == 0:
            picam2.start()
        set_preview = 1
        self.update_preview()
    
    def reset(self):
        global set_preview, picam_running,control_signal, previous_error
        dac.value = V0
        self.after_cancel(self.update_preview)
        if picam_running == 0:
            picam2.start()
        set_preview = 1
        dac.value = V0
        self.control_signal = 0
        self.previous_error = 0
        self.integral = 0
        pid.reset()
        self.update_preview()
    
    def set_zoom_4(self):
        self.zoom_ratio = 4
    
    def set_zoom_1(self):
        self.zoom_ratio = 1
    
    def set_zoom_10(self):
        self.zoom_ratio = 10    
        
        

if __name__ == "__main__":
    app = Application()
    app.eval('tk::PlaceWindow . center')
    app.mainloop()



