import cv2

# Function to read video and get frames
def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True: # runs until the video is over
        ret, frame = cap.read() # ret=True if the frame is read correctly
        if not ret:
            break
        frames.append(frame)
    cap.release
    return frames

# Function to save video
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print("Video saved to:", output_video_path)