import cv2
import torch
import numpy as np
import time
import matplotlib
import serial
from depth_anything_v2.dpt import DepthAnythingV2


SERIAL_PORT = "COM4"  
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"Error opening serial port: {e}")
    ser = None


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


ENCODER_TYPE = 'vits'
depth_anything = DepthAnythingV2(**model_configs[ENCODER_TYPE]).to(DEVICE)
depth_anything.load_state_dict(torch.load(
    'checkpoints/depth_anything_v2_vits.pth',
    map_location=DEVICE
))
depth_anything.eval()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


cmap = matplotlib.colormaps.get_cmap('Spectral_r')
font = cv2.FONT_HERSHEY_SIMPLEX
fullscreen = False
window_name = "Depth Estimation"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


rows, cols = 2, 3
matrix_result = np.zeros((rows, cols), dtype=int)
last_print_time = time.time()
last_serial_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    input_size = 518
    frame_resized = cv2.resize(frame, (input_size, input_size))
    input_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

    with torch.inference_mode():
        depth_tensor = depth_anything(input_tensor)
        depth = depth_tensor.squeeze().cpu().numpy()

    depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)
    depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    h, w = frame.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    # *Draw Grid*
    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_h), (w, i * cell_h), (0, 255, 0), 2)
        cv2.line(depth_colored, (0, i * cell_h), (w, i * cell_h), (255, 255, 255), 1)

    for j in range(1, cols):
        cv2.line(frame, (j * cell_w, 0), (j * cell_w, h), (0, 255, 0), 2)
        cv2.line(depth_colored, (j * cell_w, 0), (j * cell_w, h), (255, 255, 255), 1)

    # *Process Grid Cells*
    for r in range(rows):
        for c in range(cols):
            x_start, y_start = c * cell_w, r * cell_h
            cell_roi = depth_colored[y_start:y_start + cell_h, x_start:x_start + cell_w]
            detected_value = int(np.mean(cell_roi) / 51)
            matrix_result[r, c] = detected_value

    # *Determine Movement Direction*
    count_2s = np.count_nonzero(matrix_result == 2)
    if count_2s >= 4:
        direction = "STOP"
        serial_command = "S"
    else:
        left_avg = np.mean(matrix_result[:, :cols//2])
        right_avg = np.mean(matrix_result[:, cols//2:]) 
        
        front_row_avg = np.mean(matrix_result[0])
        middle_column_avg = np.mean(matrix_result[:, 1])
        bottom_middle = matrix_result[1][1]
        
        front_clear_threshold = 2.5
        middle_column_threshold = 2.8
        
        if front_row_avg >= 2.8 and middle_column_avg >= 2.6 and bottom_middle >= 2:
            direction = "STRAIGHT"
            serial_command = "SA" 
        elif left_avg < right_avg:
            direction = "LEFT"
            serial_command = "L"
        else:
            direction = "RIGHT"
            serial_command = "R"

    # *Send Data to Serial Every 1.5 sec*
    if ser and time.time() - last_serial_time >= 1.5:
        try:
            ser.write(f"{serial_command}\n".encode())
            print(f"Sent: {serial_command}")
            last_serial_time = time.time()
        except Exception as e:
            print(f"Serial Write Error: {e}")

    # *Display Direction UI*
    direction_overlay = frame.copy()
    cv2.rectangle(direction_overlay, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    color = (255, 255, 255) if direction != "STOP" else (0, 0, 255)
    cv2.putText(direction_overlay, f"Suggested Move: {direction}", (20, h - 20), font, 0.8, color, 2)
    cv2.addWeighted(direction_overlay, 0.6, frame, 0.4, 0, frame)

    # *Print Matrix Every Second*
    if time.time() - last_print_time >= 1:
        # print("\nMatrix Result:")
        for row in matrix_result:
            # print("  ".join(map(str, row)))  # Prints matrix in readable format
            print("")
        print(f"Suggested Move: {direction} | Sent Command: {serial_command}")
        last_print_time = time.time()

    # *Combine Views*
    split_region = np.ones((h, 50, 3), dtype=np.uint8) * 255
    combined = cv2.hconcat([frame, split_region, depth_colored])
    cv2.imshow(window_name, combined)

    # *Handle Key Presses*
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.destroyWindow(window_name)
        if fullscreen:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# *Cleanup*
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()