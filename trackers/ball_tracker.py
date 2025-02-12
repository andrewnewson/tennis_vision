from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) # load yolo model from path

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions] # get the ball positions from the ball detections
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"]) # create a dataframe from the ball positions
        df_ball_positions = df_ball_positions.interpolate() # interpolate the missing ball positions
        df_ball_positions = df_ball_positions.bfill() # backfill the missing ball positions at beginning of the video

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()] # convert the ball positions back to the original format

        return ball_positions
    
    def get_ball_hit_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions] # get the ball positions from the ball detections
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"]) # create a dataframe from the ball positions
        df_ball_positions['middle_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2 # calculate the middle y position of the ball
        df_ball_positions['middle_y_rolling_mean'] = df_ball_positions['middle_y'].rolling(window=5, min_periods=1, center=False).mean() # calculate the rolling mean of the middle y position of the ball
        df_ball_positions['delta_y'] = df_ball_positions['middle_y_rolling_mean'].diff() # calculate the difference between the rolling mean of the middle y position of the ball
        df_ball_positions['ball_hit'] = 0 # initialise a new column to store whether the ball has been hit
        minimum_change_frames_for_hit = 25 # set the minimum number of frames for a hit to be detected
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit*1.2)): # iterate through each frame
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0 # check if the ball is moving upwards
            positive_postion_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0 # check if the ball is moving downwards

            if negative_position_change or positive_postion_change: # check if the ball is moving upwards or downwards
                change_count = 0
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1): # iterate through the next frames
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0 # check if the ball is moving upwards in the next frame
                    positive_postion_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0 # check if the ball is moving downwards in the next frame

                    if negative_position_change and negative_position_change_following_frame: # check if the ball is moving upwards in the next frame
                        change_count += 1
                    elif positive_postion_change and positive_postion_change_following_frame: # check if the ball is moving downwards in the next frame
                        change_count += 1
                
                if change_count >= minimum_change_frames_for_hit-1: # check if the ball has moved upwards or downwards for the minimum number of frames
                    df_ball_positions.loc[i, 'ball_hit'] = 1 # set the ball hit column to 1 (avoiding setcopy warning)

        hit_frame_numbers = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist() # get the frame numbers where the ball has been hit

        return hit_frame_numbers

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames: # iterate through each video frame
            ball_dict = self.detect_frame(frame) # detect ball in the frame
            ball_detections.append(ball_dict) # append the ball detections to the list

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections
    
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0] # run object detection on the frame

        ball_dict = {}
        for box in results.boxes: # iterate through each detected box
            result = box.xyxy.tolist()[0] # get the bounding box coordinates
            ball_dict[1] = result # add the bounding box coordinates to the ball dictionary

        return ball_dict
    
    def draw_bounding_boxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections): # iterate through each frame and ball detection
            for track_id, bbox in ball_dict.items(): # iterate through each ball detection
                x1, y1, x2, y2 = bbox # get the bounding box coordinates
                cv2.putText(frame, f"Ball ID {track_id}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2) # add ball ID text to the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2) # draw bounding box around ball
            output_video_frames.append(frame) # append the frame with bounding boxes to the output list

        return output_video_frames