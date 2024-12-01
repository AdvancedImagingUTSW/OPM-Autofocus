
# Active Remote Focus Stabilization in Oblique Plane Microscopy

This repository contains Python scripts and tools developed for an **Active Remote Focus Stabilization System** designed for **Oblique Plane Microscopy (OPM)**. The system uses a Raspberry Pi to implement a real-time feedback control mechanism, ensuring sub-nanometer precision for long-term imaging without interrupting fluorescence imaging. This stabilization approach enables high-resolution imaging of subcellular structures over extended periods.

## Project Overview

### Background
Oblique Plane Microscopy (OPM) is a light-sheet fluorescence microscopy (LSFM) variant that uses a single primary objective for both illumination and detection. Despite its advantages, the remote focusing unit in OPM is prone to axial drift, causing imaging degradation. The stabilization system in this repository solves this problem by using a laser alignment beam to actively monitor and correct the focal drift of secondary and tertiary objectives in real-time. 

The core components:
1. A Raspberry Pi system with a high-speed Pi Camera (PiCam).
2. A proportional-integral-derivative (PID) control algorithm for axial correction.
3. A graphical user interface (GUI) for real-time monitoring and parameter tuning.

### Features
- **Autofocus Calibration and Analysis**: Creates calibration curves and analyzes the stabilization performance.
- **Real-Time Autofocus Control**: Continuously monitors the focal plane and applies corrections via a piezoelectric actuator.
- **Customizable GUI**: Intuitive interface for users to adjust system parameters and monitor performance.

---

## Repository Structure

### Files
1. **`OPM-Autofocus_Analysis_and_Visualize_Calibration_Curve.py`**
   - Generates calibration curves correlating piezo voltage to focal shifts.
   - Provides analysis tools for evaluating stabilization performance.

2. **`OPM-Autofocus_GUI_on_RasberryPi.py`**
   - Implements the autofocus system using a Raspberry Pi and PiCam.
   - Includes a PID controller for real-time axial drift compensation.
   - Offers a GUI for parameter tuning and live error visualization.

---

## Installation

### Prerequisites
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

## Example Outputs

- **Calibration Curve**: 
  Plots showing the relationship between actuator voltage and focal displacement.
- **Stabilization Results**:
  - Real-time plots of laser spot position and error signals.
  - Logs of actuator adjustments.

---

## References

- Nguyen, T. D., Rahmani, A., Ponjavic, A., Millett-Sikking, A., & Fiolka, R. *Active Remote Focus Stabilization in Oblique Plane Microscopy*. (bioRxiv.org - Comming soon)


---

## Acknowledgments

We would like to thank Dr. Vasanth Siruvallur Murali for preparing the A375 cancer cells. R.F. is thankful for support from the National Institute of Biomedical Imaging and Bioengineering (grant R01EB035538) and the National Institute of General Medical Sciences (grant R35GM133522).

