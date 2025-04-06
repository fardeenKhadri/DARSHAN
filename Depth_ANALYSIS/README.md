# DepthTheView

## Overview

DepthTheView is a real-time monocular depth estimation tool that utilizes your webcam to detect obstacles within a specified region. When an object is detected within a certain distance, the system alerts the user with a sound notification.

## Features

- **Real-Time Depth Estimation**: Continuously analyzes webcam feed to compute depth information.
- **Obstacle Detection**: Identifies objects within a user-defined area and distance threshold.
- **Audio Alerts**: Plays a sound notification when an obstacle is detected within the specified parameters.

## Installation


1. **Install Dependencies**:
   Ensure you have Python installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

To start the depth estimation and obstacle detection:

```bash
python run.py
```

**Controls**:
- Press `q` to exit the application.

## Configuration

- **Detection Box Size and Position**: Adjust the `BOX_WIDTH` and `BOX_HEIGHT` variables in `run.py` to change the size of the detection area. The box is centered by default.
- **Depth Threshold**: Modify the `THRESHOLD_DISTANCE_METERS` variable to set the distance at which the system should trigger an alert.
- **Depth Scale Factor**: The `DEPTH_SCALE_FACTOR` is used to convert depth map values to real-world distances. Adjust this value based on calibration and testing.

## Dependencies

- `opencv-python`
- `numpy`
- `torch`
- `pygame`

Ensure all dependencies are installed using the provided `requirements.txt`.

## Notes

- The application forces CPU execution by default. If you have a compatible GPU and wish to utilize it, remove or modify the following lines in `run.py`:
  ```python
  os.environ["XFORMERS_FORCE_DISABLE"] = "1"
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend=cpu"
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  ```
  Additionally, change `DEVICE = "cpu"` to `DEVICE = "cuda"`.

- The depth estimation model is loaded from the `DepthAnything` library. Ensure the model weights are correctly downloaded and accessible.

## Acknowledgments

This project utilizes the [Depth Anything](https://github.com/LiheYoung/Depth-Anything) model for depth estimation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: Customize the paths and configurations as needed to suit your specific setup and requirements.* 
