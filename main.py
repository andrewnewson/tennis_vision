from utils import *
from trackers import *
from court_detector import *
from mini_court_graphic import *
from copy import deepcopy
import pandas as pd
import constants
import os

def main():
    # Read video
    input_video_path = select_file()
    if not input_video_path:
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

    # Filter out players that are not on the court
    player_detections = player_tracker.select_players_only(court_keypoints, player_detections)

    # Initialise mini court graphic
    mini_court_graphic = MiniCourtGraphic(video_frames[0])

    # Detect ball hits
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)

    # Convert positions to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court_graphic.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

    # Loop over ball hits to get stats
    player_stats_data = [{
        "frame_number": 0,
        "player_1_number_hits": 0,
        "player_1_total_hit_speed": 0,
        "player_1_last_hit_speed": 0,
        "player_1_total_player_speed": 0,
        "player_1_last_speed": 0,

        "player_2_number_hits": 0,
        "player_2_total_hit_speed": 0,
        "player_2_last_hit_speed": 0,
        "player_2_total_player_speed": 0,
        "player_2_last_speed": 0
    }]

    for hit_idx in range(len(ball_hit_frames)-1):
        start_frame = ball_hit_frames[hit_idx]
        end_frame = ball_hit_frames[hit_idx+1]
        ball_hit_time_secs = (end_frame - start_frame) / 24 # 24 fps video

        hit_distance_pxl = measure_distance(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        hit_distance_metres = convert_pixel_distance_to_metres(hit_distance_pxl, constants.DOUBLE_COURT_WIDTH, mini_court_graphic.get_width_of_mini_court())

        speed_of_shot = hit_distance_metres / ball_hit_time_secs * 3.6 # speed in km/h

        player_positions = player_mini_court_detections[start_frame]
        player_hit_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id], ball_mini_court_detections[start_frame][1]))

        opponent_player_id = 1 if player_hit_ball == 2 else 2
        distance_covered_by_opponent_pxl = measure_distance(player_mini_court_detections[start_frame][opponent_player_id], player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_metres = convert_pixel_distance_to_metres(distance_covered_by_opponent_pxl, constants.DOUBLE_COURT_WIDTH, mini_court_graphic.get_width_of_mini_court())

        opponent_speed = distance_covered_by_opponent_metres / ball_hit_time_secs * 3.6 # speed in km/h

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_number"] = start_frame
        current_player_stats[f"player_{player_hit_ball}_number_hits"] += 1
        current_player_stats[f"player_{player_hit_ball}_total_hit_speed"] += speed_of_shot
        current_player_stats[f"player_{player_hit_ball}_last_hit_speed"] = speed_of_shot

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += opponent_speed
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = opponent_speed

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_number": list(range(len(video_frames)))})

    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, how="left")
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_avg_hit_speed"] = player_stats_data_df["player_1_total_hit_speed"] / player_stats_data_df["player_1_number_hits"]
    player_stats_data_df["player_2_avg_hit_speed"] = player_stats_data_df["player_2_total_hit_speed"] / player_stats_data_df["player_2_number_hits"]
    player_stats_data_df["player_1_avg_player_speed"] = player_stats_data_df["player_1_total_player_speed"] / player_stats_data_df["player_2_number_hits"]
    player_stats_data_df["player_2_avg_player_speed"] = player_stats_data_df["player_2_total_player_speed"] / player_stats_data_df["player_1_number_hits"]

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