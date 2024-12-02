# Active Remote Focus Stabilization in Oblique Plane Microscopy

This repository contains Python scripts and tools developed for an **Active Remote Focus Stabilization System** designed for **Oblique Plane Microscopy (OPM)**. The system uses a Raspberry Pi to implement a real time feedback control mechanism, ensuring sub-nanometer precision for long-term imaging without interrupting fluorescence imaging. This stabilization approach enables high-resolution imaging of subcellular structures over extended periods.

## Project Overview

### Background
Oblique Plane Microscopy (OPM) is a light-sheet fluorescence microscopy (LSFM) variant that uses a single primary objective for both illumination and detection. Despite its advantages, the remote focusing unit in OPM is prone to axial drift, causing imaging degradation. The stabilization system in this repository solves this problem by using a laser alignment beam to actively monitor and correct the focal drift of secondary and tertiary objectives in real time. 

The core components:
1. A Raspberry Pi system with a high-speed Pi Camera (PiCam).
2. A proportional-integral-derivative (PID) control algorithm for axial correction.
3. A graphical user interface (GUI) for real time monitoring and parameter tuning.

### Features
- **Autofocus Calibration and Analysis**: Creates calibration curves and analyzes the stabilization performance.
- **Real-time Autofocus Control**: Continuously monitors the focal plane and applies corrections via a piezoelectric actuator.
- **Customizable GUI**: Intuitive user interface to adjust system parameters and monitor performance.

---

## Repository Structure

### Files
1. **`OPM-Autofocus_Analysis_and_Visualize_Calibration_Curve.py`**
   - Generates calibration curves correlating piezo voltage to focal shifts.
   - Provides analysis tools for evaluating stabilization performance.

2. **`OPM-Autofocus_GUI_on_RasberryPi.py`**
   - Implements the autofocus system using a Raspberry Pi and PiCam.
   - Includes a PID controller for real time axial drift compensation.
   - Offers a GUI for parameter tuning and live error visualization.

---

## Installation

### Prerequisites
- **Optics**:

See our bioRxiv paper [1] (https://doi.org/10.1101/2024.11.29.626121) for full optical setup instructions.

For illumination, laser light from a previously published light-sheet module [2] is coupled over a dichroic mirror (Di03-R405/488/561/635-t1-25x36, Semrock) into the optical train of the OPM. After passing through two tube lenses (TL 1: ITL-200, TL2: custom, 150mm EFL) and two galvo mirrors (Pangolin Saturn 9B with a 10mm y-mirror mounted 90 degrees rotated) for image space scanning [3], an oblique sheet is launched in sample space through the primary objective (O1, Nikon 40X, NA 1.25 silicone oil). 

Fluorescence light is detected through the same objective, whose pupil is conjugated to the secondary objective (O2, Nikon 40X NA 0.95). The secondary objective maps fluorescence emitters along the light-sheet into the remote space while minimizing spherical aberrations. A tertiary imaging system is used to map the fluorescence on a scientific CMOS camera (Orca Flash 4, Hamamatsu). The tertiary imaging system consists of a glass-tipped objective (AMS-AGY v2 54-18-9, Applied Scientific instrumentation) and a tube lens (f=300mm achromatic doublet, Thorlabs).
  
![OPM-Autofocus-Fig 1](https://github.com/user-attachments/assets/c11085e2-f4e3-436b-ac8d-73e49a8d9c0a)

Fig. 1. Schematic layout of an oblique plane microscope with remote focus stabilization. **A** Schematic layout of the OPM. Green shows the fluorescence path, blue shows the light-sheet path and magenta shows the alignment laser path. O1: primary objective, O2: secondary objective, O3: tertiary objective. TL1-TL3: first, second and third tube lens. **B** Position of the alignment laser beam (blue) in the pupil of the secondary objective O2. **C** Laser focus in the remote imaging space between O2 and O3. **D** Laser focus shown in an orthogonal plane to **C**. Dotted lines correspond to the focal plane of O3.

For the remote focus stabilization, we injected a collimated laser beam over the backside of the dichroic mirror into the optical train of the OPM (Figure 1A). For the experiments shown here, we used a fiber coupled 488nm laser (Edmund Optics, 10mW Pigtailed Laser Diode, part nr: #23-761), which was collimated with a f=100mm lens. The reflective side of the dichroic faces towards the primary objective (i.e. the light-sheet illumination laser bounces off the coated front surface of the dichroic). In contrast, the alignment laser first travels through the dichroic mirror substrate. As such, there is a secondary reflection from the substrate itself. We reasoned that if we use a laser line for which the dichroic was optimized for reflection, the secondary reflection would be minor. Indeed, we observed a notable double reflection when using another wavelength (i.e. laser diode at 785nm), whereas the 488nm line showed a single dominant reflection.

The alignment laser beam passes through the secondary and tertiary objective and is picked up by a dichroic mirror (Semrock Di02-R488-25x36) after the tertiary tube lens. The laser beam is then focused on a camera (OV9281-120, labeled PiCam in Figure 1A) used for the focus stabilization feedback. While it is advisable to put filters and mirrors in the infinity space, we found that with a tube lens of low optical power, placing the dichroic in the “image space” causes negligible aberrations. Adopters of the technology may consider placing the dichroic in the infinity space of the tertiary imaging system when using a tube lens with shorter focal length.

Our system measures the relative misalignment of the remote focus system by using a laser beam that is tilted to the optical axes of the secondary and tertiary objective (labeled Z and Z’ respectively). As such, an axial misalignment not only causes the laser beam to defocus, but also to be translated on the alignment camera. To this end, we inject the alignment beam at an off-center position into the secondary pupil (Figure 1B). This causes the beam to tilt in the remote focus space, as shown in Figure 1C. As such, if either the secondary or tertiary objective is shifted axially, i.e. its focal plane moves, the beam is defocused, but also translated laterally along the Y’ axis on the alignment camera.

In the other dimension, a tilt of the laser beam occurs naturally, as the tertiary imaging system is angled to the optical axis of the secondary objective (Figure 1D). In case of focal drift of the tertiary objective, the laser beam gets translated in the X’ direction, whereas a drift from the secondary objective will not show in the X’ direction (see also Figure 1D for an illustration).

The off-centering of the alignment laser in the O2 pupil increases the sensitivity of the measurement (a given focal shift causes a larger lateral shift on the camera), but it also ensures that axial shifts of both O2 and O3 become measurable. To place the laser beam off-center in the pupil of O2, its numerical aperture has to be reduced (i.e. the laser beam underfills the pupil). This in turn increases the depth of focus of the laser beam and hence makes fitting of the laser spot easier over a larger defocus range. Importantly, it does not lower the localization precision for the scenario of “unlimited photons”. While the spot gets larger in size when reducing its NA, more photons (if the laser intensity of the laser beam is increased accordingly) contribute to the measurement, which restores the measurement precision.

- **Hardware**:
  - Raspberry Pi 4B or later.
  - Raspberry Pi Camera (OV9281 or compatible high-speed camera).
  - Piezo actuator (e.g., Thorlabs PC4GR) and associated driver.
  - Optical setup as described in the manuscript (laser, lenses, objectives).

- **Software**:
  - Python 3.8 or later.
  - Required Python libraries:
    ```
    pip install numpy scipy matplotlib opencv-python PyQt5
    ```

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/AdvancedImagingUTSW/OPM-Autofocus.git
   cd opm-autofocus
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Calibration and Analysis**
   - Run the calibration script to generate a calibration curve:
     ```bash
     python OPM-Autofocus_Analysis_and_Visualize_Calibration_Curve.py
     ```
   - Follow on-screen instructions to input actuator displacement data.
   - The script outputs:
     - Calibration curve plots.
     - Drift compensation analysis.

### 2. **Autofocus System**
   - Launch the autofocus GUI:
     ```bash
     python OPM-Autofocus_GUI_on_RasberryPi.py
     ```
   - **Steps**:
     1. **Initial Calibration**: Align the laser spot and set the target position.
     2. **PID Tuning**: Adjust proportional and integral gains for optimal stabilization.
     3. **Start Monitoring**: Visualize real-time laser spot position and error signal.

---

## How It Works

1. **Laser Alignment**:
   - A collimated laser beam is injected into the optical path, passing through the secondary and tertiary objectives.
   - The beam's position is detected by the PiCam, capturing lateral shifts induced by focal drift.

2. **Error Signal Generation**:
   - The position of the laser spot is extracted via Gaussian fitting on the x and y axes of the PiCam images.
   - Deviations from the calibrated setpoint are computed as an error signal.

3. **Feedback Control**:
   - A PID controller computes corrective signals based on the error.
   - These signals adjust the piezo actuator's position, maintaining alignment.

---

## Graphical User Interface (GUI) for Autofocus Control System *(OPM-Autofocus_GUI_on_RasberryPi.py)*

The GUI designed for the autofocus system serves as an interactive platform for setting parameters, monitoring performance, and visualizing the point spread function (PSF) in real-time. It is built using the Tkinter library in Python and is optimized for user-friendly operation with a touchscreen. This appendix provides a detailed description of the GUI layout and the functions of each component.

The GUI designed for the autofocus system serves as an interactive platform for setting parameters, monitoring performance, and visualizing the point spread function (PSF) in real-time. It is built using the Tkinter library in Python and is optimized for user-friendly operation with a touchscreen. This section provides a detailed description of the GUI layout and the functions of each component.

**1\. Main Layout and Frames:**

The GUI is organized into multiple sections for clarity and functional grouping. The groups are dynamically sized to accommodate user inputs and visual feedback.

![OPM-Autofocus-Fig 4](https://github.com/user-attachments/assets/a7293962-f43c-406e-bc62-4510a4da91f7)

Fig. S1. Graphical User Interface (GUI) for Autofocus Control System.

**a) Image Preview Group (blue box in the Fig. S1)**

This group provides a live feed of the camera view, enabling visualization of the PSF in real time.

Image Canvas:  
A canvas widget where the live feed from the PiCam is displayed. This feed includes the real-time PSF overlays with the calculated center and setpoint for quick verification.

Zoom Controls:  
1x 4x and 10x buttons to zoom in on a region of interest (ROI) for detailed observation of the PSF

**b) Control Parameters Group (red box in the Fig. S1)**

This group is dedicated to parameter input for the autofocus system. It allows users to fine-tune the PID control loop and other system settings.

Proportional Gain (P):  
A text box where users can input the proportional gain (Kp) for the PID controller. This controls the immediate response to error.

Integral Gain (I):  
A text box for entering the integral gain (Ki), which addresses cumulative error over time.

Derivative Gain (D):  
A text box for setting the derivative gain (Kd). Although Kd is initially set to zero, this option allows flexibility for future adjustments

**c) System Calibration Frame (green box in the Fig. S1)**

This group allows users to calibrate the system before starting an experiment.

Calib Button:  
A button to initialize the laser spot's reference position (setpoint). This ensures accurate error calculations throughout the session.

Reset Button:  
A button to reset the calibration and clear the setpoint, allowing for a new reference point to be defined.

Start/Stop Buttons:

Buttons to start/stop the PID autofocus loop.

Preview Button:

A button to toggle the preview function.

**2\. Interaction Workflow:**

**Setup**:

The user inputs PID gains in the **Control Parameters Group**.

The **Calibration Button** in the **System Calibration Group** is pressed to set the initial position of the laser spot.

**Operation**:

The real-time feed in the **Image Preview Group** provides visual feedback of the PSF.

**Adjustments**:

Parameters can be updated: click **Stop Button** -> adjust PID gains in the **Control Parameters Group** -> click **Start Button**.

The user can recalibrate the system if necessary: click **Stop Button** -> click **Reset Button** and repeat the **Setup** process to set a reference position (setpoint).

---

## Autofocus Calibration and Analysis *(OPM-Autofocus_Analysis_and_Visualize_Calibration_Curve.py)*

The developed GUI provides an interactive tool for processing and analyzing PSF (point spread function) images acquired during calibration of the autofocusing system. The software takes input images captured as the piezo stack moves the tertiary objective lens from -2.5 µm to +2.5 µm in fine increments. These images serve as the basis for generating calibration curves and computing error functions, with the PSF position measured at each displacement.

![OPM-Autofocus-Fig S2](https://github.com/user-attachments/assets/6a9e27c4-c72c-4964-a508-e073cba10ccc)

Fig. S2. Autofocus calibration Curve and Analysis. 

**Step-by-Step Usage Instructions:**

**1\. Launch the Application**:

Run the Python script to open the GUI. The interface is divided into sections for selecting input files, specifying an output folder, applying temporal color coding, and visualizing results.

**2\. Select PSF Image Files**:

Click the **"Select Files"** button to open a file dialog.

Navigate to the folder containing the PSF images and select the entire series of images corresponding to the piezo stack displacements. The selected files will be listed, and the number of images will be displayed for confirmation.

**3\. Set the Output Folder**:

Click the **"Select Folder"** button to choose the folder where processed results will be saved.

**4\. Choose a Colormap for Temporal Color Coding**:

Select a colormap from the dropdown menu in the "Colormap" section. The colormap is used to apply temporal color coding to represent different axial (z-axis) displacements in the input images.

Adjustments to the colormap automatically reset the GUI for real-time experimentation with visualization styles.

**5\. Apply Temporal Color Coding**:

Press the **"Apply Temporal Color Coding"** button to process the selected images.

The software converts grayscale images into temporally color-coded representations, where colors correspond to specific z-displacements. The processed images are saved as a stack compatible with ImageJ for further analysis.

Maximum intensity projection (MIP) images with and without a color bar are automatically generated and saved in the specified output folder.

**6\. View and Analyze Results**:

The **MIP (Maximum Intensity Projection)** of the processed images is displayed in the GUI under the "MIP" section. This provides an overview of the axial intensity distribution.

The "Video" section visualizes the temporal progression of the processed images.

**7\. Save Individual Frames**:

Use the **"Save Current Frame"** button to save the currently displayed frame from the video section to the output folder for closer examination.

**8\. Play and Pause Processed Video**:

Start the video playback by clicking **"Start Video"**. The processed image stack is displayed as a loop, illustrating temporal color coding over the z-displacement range.

Use the **"Stop Video"** button to pause playback.

---

## Example Outputs

- **Calibration Curve**: 
  Plots showing the relationship between error function and focal displacement.

To measure the accuracy of our stabilization method, we acquired a rapid timelapse sequence, with a duration of 100 seconds to minimize drift. The standard deviation of the localization measurement of the laser spot was then converted into nanometers using a calibration curve (where the piezo stack was stepped in known increments), as shown in Figure 2A-C and Figure 3A. We measured a standard deviation of ~57 nm using this approach. If converted back to the sample space (demagnification of 1.33), this corresponds to an axial precision of 43 nm, which is an order of magnitude lower than the depth of focus of the detection system.

![OPM-Autofocus-Fig 2](https://github.com/user-attachments/assets/c25af649-1fa7-4bac-8ac8-7928cceb6cb9)

Fig. 2. Axial drift estimation using spatial displacement. A X’ and Y’ displacement of the laser PSF extracted from tertiary objective Z’-scan. B Calibration curve of focal drift estimation using Euclidean Distance. C Z’-position color-coded projection image of 300 laser spots captured from a tertiary objective scan, insets show the PSFs at -1.5-micron, 0 micron and +1.5 microns. The laser PSF location moves diagonally over the image sensor through the Z’-scan.

- **Stabilization Results**:
  - Real-time plots of laser spot position and error signals.

We then compared the long-term stability of the remote focusing system with and without focus stabilization. Without feedback correction, the system drifted ~2 microns over an hour (Figure 3B). This is not unexpected, as our laboratory experiences temperature oscillations with an amplitude of ~1 degree Celsius (see also Appendix for temperature measurements). With the focus stabilization on, the standard deviation over an hour was 150 nm measured in the remote space, corresponding to 113 nm mapped into sample space.

![OPM-Autofocus-Fig 3](https://github.com/user-attachments/assets/57e7f752-847f-4534-a06f-8099c5d6ffed)

Fig. 3. A Time-lapse acquisition at the focal plane for estimating axial precision. B Axial drift with stabilization ON (red) and OFF (black) monitored for 1 hour. S.D. Standard deviation. 

  - Logs of actuator adjustments.

---

## References

1. Nguyen, T. D., Rahmani, A., Ponjavic, A., Millett-Sikking, A., & Fiolka, R. (2024). **Active Remote Focus Stabilization in Oblique Plane Microscopy**. bioRxiv (https://doi.org/10.1101/2024.11.29.626121). 
2. Chen, B., Chang, B.-J., Zhou, F. Y., Daetwyler, S., Sapoznik, E., Nanes, B. A., Terrazas, I., Gihana, G. M., Perez Castro, L., Chan, I. S., Conacci-Sorrell, M., Dean, K. M., Millett-Sikking, A., York, A. G., & Fiolka, R. (2022). Increasing the field-of-view in oblique plane microscopy via optical tiling. Biomed. Opt. Express, 13, 5616–5627.
3. Daetwyler, S., Chang, B.-J., Chen, B., Voigt, F. F., Rajendran, D., Zhou, F., & Fiolka, R. (2023). Mesoscopic oblique plane microscopy via light-sheet mirroring. Optica, 10, 1571–1581.



---

## Acknowledgments

We would like to thank Dr. Vasanth Siruvallur Murali for preparing the A375 cancer cells. R.F. is thankful for support from the National Institute of Biomedical Imaging and Bioengineering (grant R01EB035538) and the National Institute of General Medical Sciences (grant R35GM133522).

