# Initialize the global variable to store the last 3 frames
last_3_frames = []

def save_frame(frame):
    global last_3_frames
    last_3_frames.append(frame)
    if len(last_3_frames) > 3:  # Keep only the last 3 frames
        last_3_frames.pop(0)

def get_last_frames():
    global last_3_frames
    if last_3_frames:  # Check if the list is not empty
        return list(last_3_frames)
    else:
        return None