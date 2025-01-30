import pandas as pd
import sys
sys.path.append("../")
from utils import *
import constants
from mini_court_graphic import *
from copy import deepcopy

def calc_basic_player_stats(ball_hit_frames, player_mini_court_detections, ball_mini_court_detections, video_frames, mini_court_graphic):
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

    return player_stats_data_df