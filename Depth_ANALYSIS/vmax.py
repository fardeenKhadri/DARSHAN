import cv2
import torch
import numpy as np
import time
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

# Set CUDA Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# Model Configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Load Model on CUDA
ENCODER_TYPE = 'vits'
depth_anything = DepthAnythingV2(**model_configs[ENCODER_TYPE]).to(DEVICE)
depth_anything.load_state_dict(torch.load(
    'checkpoints\depth_anything_v2_vits.pth',
    map_location=DEVICE
))
depth_anything.eval()

# Webcam Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# UI Elements
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
font = cv2.FONT_HERSHEY_SIMPLEX
fullscreen = False  # Track fullscreen state
window_name = "Depth Estimation"

# Initial Window Creation
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Grid Config
rows, cols = 2, 3
matrix_result = np.zeros((rows, cols), dtype=int)
last_print_time = time.time()

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize Frame
    input_size = 518
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # Convert to Tensor and Move to CUDA
    input_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

    # Model Inference
    with torch.inference_mode():
        depth_tensor = depth_anything(input_tensor)
        depth = depth_tensor.squeeze().cpu().numpy()

    # Normalize Depth Map
    depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)

    # Apply Colormap
    depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # Frame Dimensions
    h, w = frame.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    # Draw Grid
    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_h), (w, i * cell_h), (0, 255, 0), 2)
        cv2.line(depth_colored, (0, i * cell_h), (w, i * cell_h), (255, 255, 255), 1)

    for j in range(1, cols):
        cv2.line(frame, (j * cell_w, 0), (j * cell_w, h), (0, 255, 0), 2)
        cv2.line(depth_colored, (j * cell_w, 0), (j * cell_w, h), (255, 255, 255), 1)

    # UI Overlay (Title Bar)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 50), (0, 0, 0), -1)  
    cv2.putText(overlay, "D.A.R.S.H.A.N - Press 'Q' to Exit | 'F' to Toggle Fullscreen",
                (20, 40), font, 0.6, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Process Grid Cells
    for r in range(rows):
        for c in range(cols):
            x_start, y_start = c * cell_w, r * cell_h
            cell_roi = depth_colored[y_start:y_start + cell_h, x_start:x_start + cell_w]

            detected_value = int(np.mean(cell_roi) / 51)  # Normalized depth
            matrix_result[r, c] = detected_value

    # Determine Movement Direction
    left_avg = np.mean(matrix_result[:, :cols//2])
    right_avg = np.mean(matrix_result[:, cols//2:])
    if matrix_result[0, 1] >= 4 and matrix_result[1, 1] >= 4:
        direction = "STRAIGHT"
    else:
        direction = "LEFT" if left_avg < right_avg else "RIGHT"

    # Display Direction UI
    direction_overlay = frame.copy()
    cv2.rectangle(direction_overlay, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.putText(direction_overlay, f"Suggested Move: {direction}", (20, h - 20),
                font, 0.8, (255, 255, 255), 2)
    cv2.addWeighted(direction_overlay, 0.6, frame, 0.4, 0, frame)

    # Print Matrix Every Second
    if time.time() - last_print_time >= 1:
        # print("\nMatrix Result:")
        # print(matrix_result)
        print(f"{direction}")
        last_print_time = time.time()

    # Combine Views
    split_region = np.ones((h, 50, 3), dtype=np.uint8) * 255
    combined = cv2.hconcat([frame, split_region, depth_colored])

    # Display Window
    cv2.imshow(window_name, combined)

    # Handle Key Presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # Toggle Fullscreen
        fullscreen = not fullscreen
        cv2.destroyWindow(window_name)  # Close current window
        if fullscreen:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Cleanup
cap.release()
cv2.destroyAllWindows()
