from utils import *
from trackers import *
from court_detector import *

def main():
    # Read video
    input_video_path = "input_media/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players and ball
    player_tracker = PlayerTracker(model_path="yolov8x.pt")
    ball_tracker = BallTracker(model_path="models/yolov5_ball_best.pt")

    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # Detect court lines
    court_model_path = "models/court_keypoints_model.pth"
    court_detector = CourtDetector(model_path=court_model_path)
    court_keypoints = court_detector.predict(video_frames[0])

    # Draw bounding boxes around players and ball
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    # Draw court lines
    output_video_frames = court_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Add frame number to video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Save video with player detections and bounding boxes overlay
    save_video(output_video_frames, "output_media/output_video.avi")

if __name__ == "__main__":
    main()