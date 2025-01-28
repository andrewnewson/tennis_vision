from utils import *
from trackers import *

def main():
    # Read video
    input_video_path = "input_media/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players and ball
    player_tracker = PlayerTracker(model_path="yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/yolov5_ball_best.pt")

    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    # Draw bounding boxes around players and ball
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(video_frames, ball_detections)

    # Save video with player detections and bounding boxes overlay
    save_video(output_video_frames, "output_media/output_video.avi")

if __name__ == "__main__":
    main()