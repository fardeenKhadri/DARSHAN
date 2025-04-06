
# DARSHAN: Dynamic AI for Real-time Sight, Haptic Assistance, and Navigation

DARSHAN is a project aimed at developing an assistive system that leverages computer vision and haptic feedback to aid visually impaired individuals in navigating their environment safely and effectively.

## Project Overview

The primary objective of DARSHAN is to integrate vision-based technologies with haptic feedback mechanisms to create a navigation aid for the visually impaired. By utilizing depth perception algorithms and real-time obstacle detection, the system provides users with tactile cues, enhancing their spatial awareness and mobility.

## Features

- **Depth Perception**: Employs advanced depth estimation techniques to understand the surrounding environment.
- **Real-time Obstacle Detection**: Identifies obstacles in the user's path and conveys information through haptic feedback.
- **Arduino Integration**: Utilizes Arduino-based hardware to process sensor data and control haptic actuators.

## Repository Structure

- `Depth_Anything/`: Contains modules and scripts related to depth estimation and processing.
- `ESP_Arduino_code/`: Houses the Arduino firmware and related code for sensor integration and haptic feedback control.

## Technologies Used

- **Programming Languages**: C++, Python, C
- **Hardware**: Arduino microcontrollers
- **Software Libraries**: OpenCV for computer vision tasks, Arduino libraries for hardware interfacing

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/fardeenKhadri/DARSHAN.git
   ```


2. **Set Up the Environment**:

   - Install the required Python packages:

     ```bash
     pip install -r requirements.txt
     ```

   - Ensure the Arduino IDE is installed for uploading code to the microcontroller.



3. **Run the Application**:

   - Execute the main script to start the vision processing and haptic feedback system:

     ```bash
     python main.py
     ```

To effectively run the DARSHAN model from the [fardeenKhadri/DARSHAN](https://github.com/fardeenKhadri/DARSHAN) repository, follow these steps:

**1. Clone the Repository:**

Begin by cloning the repository to your local machine:


```bash
git clone https://github.com/fardeenKhadri/DARSHAN.git
```


**2. Navigate to the Project Directory:**

Move into the project's root directory:


```bash
cd DARSHAN
```


**3. Set Up the Python Environment:**

It's advisable to use a virtual environment to manage dependencies:


```bash
# Create a virtual environment named 'darshan_env'
python3 -m venv darshan_env

# Activate the virtual environment
# On Windows:
darshan_env\Scripts\activate

# On macOS and Linux:
source darshan_env/bin/activate
```


**4. Install Required Dependencies:**

Install the necessary Python packages:


```bash
pip install -r requirements.txt
```


**5. Configure Hardware Components:**

The DARSHAN system integrates hardware components for haptic feedback. Ensure the following:

- **Arduino Setup:** Connect the Arduino microcontroller to your system. Upload the firmware located in the `ESP_Arduino_code` directory using the Arduino IDE.

- **Sensor Integration:** Attach the required sensors (e.g., depth sensors, cameras) as specified in the project's documentation or schematics.

**6. Run the Main Application:**

With the environment set up and hardware configured:


```bash
python main.py
```


**7. Interact with the System:**

Once the application is running:

- The system will process real-time data from the connected sensors.

- Haptic feedback will be provided based on the processed information to assist in navigation.

**8. Troubleshooting:**

If you encounter issues:

- Verify all hardware connections and ensure drivers are correctly installed.

- Check the console output for error messages and consult the project's documentation or source code for insights.

- Ensure that all dependencies are correctly installed and compatible with your system.

By following these steps, you should be able to set up and run the DARSHAN model effectively.


## Contributing

Contributions to DARSHAN are welcome. Please fork the repository and submit pull requests for any enhancements or bug fixes.


