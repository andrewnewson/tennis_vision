from utils import *
from trackers import *
from court_detector import *
from mini_court_graphic import *
from analysis import *
import os

def main():
    # Read video
    # input_video_path = select_file()
    # if not input_video_path:
    input_video_path = "input_media/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players and ball
    player_tracker = PlayerTracker(model_path="models/yolo11x.pt")
    ball_tracker = BallTracker(model_path="models/yolov5_ball_best.pt") # yolov5 best best model

    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # Detect court lines
    court_model_path = "models/court_keypoints_model.pth"
    court_detector = CourtDetector(model_path=court_model_path)
    court_keypoints = court_detector.predict(video_frames[0])

    # Filter out players that are not on the court
    player_detections = player_tracker.select_players_only(court_keypoints, player_detections)

    # Initialise mini court graphic
    mini_court_graphic = MiniCourtGraphic(video_frames[0])

    # Detect ball hits
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)

    # Convert positions to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court_graphic.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    # Calc player stats
    player_stats_data_df = calc_basic_player_stats(ball_hit_frames, player_mini_court_detections, ball_mini_court_detections, video_frames, mini_court_graphic)

    # Draw bounding boxes around players and ball
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    # Draw court lines
    output_video_frames = court_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw mini court graphic
    output_video_frames = mini_court_graphic.draw_mini_court(output_video_frames)
    output_video_frames = mini_court_graphic.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court_graphic.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, colour=(255, 0, 255))

    # Draw player stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Add frame number to video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Save video with player detections and bounding boxes overlay
    _, file_name = os.path.split(input_video_path)
    name, _ = os.path.splitext(file_name)
    output_video_path = f"output_media/{name}_output.avi"
    save_video(output_video_frames, output_video_path)

    # Save player stats df
    output_stats_path = f"output_media/{name}_stats.csv"
    player_stats_data_df.to_csv(output_stats_path, index=False)
    print("Stats saved to:", output_stats_path)

if __name__ == "__main__":
    main()